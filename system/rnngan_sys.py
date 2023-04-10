import random
import random
import sys

import torch
import yaml
from tqdm import tqdm

from system.gan_sys import TrainGAN
from util.general_util import recursive_wrap_data


class TrainRNNGANPretrain(TrainGAN):
    def run_train(self):
        self.init_train_state()
        control_dict = {'save_model_next_epoch': False}
        with open(self.control_file_path, 'w') as f:
            yaml.dump(control_dict, f)

        # GENERATOR MLE TRAINING
        print('Starting Generator MLE Training...')
        for epoch in range(self.config_dict['train_arg']['generator_pretrain_epoch']):
            print('epoch %d : ' % (epoch + 1), end='')
            self.pretrain_generator()
        self.save_model(-16)

        # torch.save(gen.state_dict(), pretrained_gen_path)
        # gen.load_state_dict(torch.load(pretrained_gen_path))

        # PRETRAIN DISCRIMINATOR
        print('\nStarting Discriminator Training...')
        for epoch in range(self.config_dict['train_arg']['discriminator_pretrain_epoch']):
            print('epoch %d : ' % (epoch + 1), end='')
            self.train_discriminator()

        # torch.save(dis.state_dict(), pretrained_dis_path)
        # dis.load_state_dict(torch.load(pretrained_dis_path))

        # # ADVERSARIAL TRAINING
        print('\nStarting Adversarial Training...')

        for epoch in range(self.epoch):
            print('\n--------\nEPOCH %d\n--------' % (epoch + 1))
            if self.config_dict['train_arg']['adaptive_adv_train']:
                while self.train_generator() > 0.6:
                    pass
                while self.train_discriminator() < 0.6:
                    pass
            else:
                # TRAIN GENERATOR
                print('\nAdversarial Training Generator : ', end='')
                for i in range(self.config_dict['train_arg']['adv_generator_epoch']):
                    self.train_generator()

                # TRAIN DISCRIMINATOR
                print('\nAdversarial Training Discriminator : ')
                for i in range(self.config_dict['train_arg']['adv_discriminator_epoch']):
                    self.train_discriminator()
                # if (epoch + 1) % self.model_save_step == 0:
            self.save_model(epoch, (0, ))

        self.save_model(-1)
        # self.save_properties()

    def pretrain_generator(self):
        """
        Max Likelihood Pretraining for the generator
        """
        optimizer_G = self.optimizer[0]
        # loss_G, loss_D = self.loss
        gen = self.model[0]
        total_loss = 0
        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            batch_size = cond_data.shape[0]
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            # print(real_gen_output)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1], device=cond_data.device, dtype=torch.long), real_gen_output[:, :-1]], dim=1)

            optimizer_G.zero_grad()
            loss, h_gen = gen.batchNLLLoss(cond_data, real_gen_output_as_input, real_gen_output)
            loss.backward()
            optimizer_G.step()

            total_loss += loss.data.item()

        sys.stdout.flush()
        print('gen.sample(5)')
        print(gen.sample(cond_data)[0][:1].cpu().detach().numpy().tolist())
        # each loss in a batch is loss per sample
        total_loss = total_loss / len(self.train_iter.dataset)
        # # sample from generator and compute oracle NLL
        # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
        #                                            start_letter=START_LETTER, gpu=CUDA)

        print(' average_train_NLL = %.4f' % total_loss)
        # print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))

    def train_generator(self):
        """
        The generator is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        optimizer_G, optimizer_D = self.optimizer[1], self.optimizer[2]
        # loss_G, loss_D = self.loss
        gen, dis = self.model
        epoch_dis_acc = 0
        epoch_pg_loss = 0
        total_sample_num = 0
        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            batch_size = cond_data.shape[0]
            total_sample_num += batch_size
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1], device=cond_data.device, dtype=torch.long), real_gen_output[:, :-1]], dim=1)

            fake, h_gen = gen.sample(cond_data)
            rewards, h_dis = dis.batchClassify(cond_data, fake)
            epoch_dis_acc += torch.sum(rewards < 0.5).data.item()

            optimizer_G.zero_grad()
            pg_loss, h_gen_PG = gen.batchPGLoss(cond_data, real_gen_output_as_input, real_gen_output, rewards)
            epoch_pg_loss += pg_loss.item()
            pg_loss.backward()
            optimizer_G.step()

        epoch_dis_acc = epoch_dis_acc / total_sample_num
        print('epoch_dis_acc %.3f' % epoch_dis_acc)

        sys.stdout.flush()
        print('gen.sample(5)')
        print(gen.sample(cond_data)[0].cpu().detach().numpy().tolist())

        return epoch_dis_acc

    def train_discriminator(self):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
        """
        optimizer_D = self.optimizer[2]
        loss_D = self.loss[1]
        gen, discriminator = self.model
        total_loss = 0
        total_acc = 0
        total_sample_num = 0

        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            if random.random() < 0.85:
                continue
            batch_size = cond_data.shape[0]
            total_sample_num += batch_size
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            # real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1]), real_gen_output[:, :-1]], dim=1)
            fake, h_gen = gen.sample(cond_data)

            optimizer_D.zero_grad()
            # fake
            out, h_dis_fake = discriminator.batchClassify(cond_data, fake)
            loss = loss_D(out, torch.zeros(batch_size, device=cond_data.device))
            total_acc += torch.sum(out < 0.5).data.item()
            # real
            out, h_dis_real = discriminator.batchClassify(cond_data, real_gen_output)
            loss += loss_D(out, torch.ones(batch_size, device=cond_data.device))
            total_acc += torch.sum(out > 0.5).data.item()

            loss.backward()
            optimizer_D.step()

            total_loss += loss.data.item()

        print(fake[0].cpu().detach().numpy().tolist())
        print(real_gen_output[0].cpu().detach().numpy().tolist())

        avg_loss = total_loss / total_sample_num
        avg_acc = total_acc / total_sample_num / 2

        sys.stdout.flush()
        print(' average_loss = %.4f, train_acc = %.4f' % (
            avg_loss, avg_acc))

        return avg_acc


class TrainSeqGANAdvLoss(TrainRNNGANPretrain):
    def pretrain_generator(self):
        """
        Max Likelihood Pretraining for the generator
        """
        optimizer_G = self.optimizer[0]
        # loss_G, loss_D = self.loss
        gen = self.model[0]
        total_loss = 0
        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            batch_size = cond_data.shape[0]
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            # print(real_gen_output)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            real_gen_output_as_input = torch.cat(
                [torch.tensor([[[1, 0.5, 0.5]] for _ in range(batch_size)], device=cond_data.device, dtype=torch.float),
                 real_gen_output[:, :-1]], dim=1
            )

            optimizer_G.zero_grad()
            loss, h_gen = gen.batchNLLLoss(cond_data, real_gen_output_as_input, real_gen_output)
            loss.backward()
            optimizer_G.step()

            total_loss += loss.data.item()

        sys.stdout.flush()
        print('gen.sample(5)')
        label, pos = gen.sample(cond_data)[0]
        print(label[0].cpu().detach().numpy().tolist())
        print(pos[0].cpu().detach().numpy().tolist()[:15])
        # each loss in a batch is loss per sample
        total_loss = total_loss / len(self.train_iter.dataset)
        # # sample from generator and compute oracle NLL
        # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
        #                                            start_letter=START_LETTER, gpu=CUDA)

        print(' average_train_loss = %.4f' % total_loss)
        # print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))
        return total_loss

    def train_generator(self):
        """
        The generator is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        optimizer_G, optimizer_D = self.optimizer[1], self.optimizer[2]
        loss_G = self.loss[0]
        gen, dis = self.model
        epoch_pg_loss, epoch_adv_loss = 0, 0
        epoch_dis_acc = 0
        total_sample_num = 0

        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            batch_size = cond_data.shape[0]
            total_sample_num += batch_size
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            real_gen_output_as_input = torch.cat(
                [torch.tensor([[[1, 0.5, 0.5]] for _ in range(batch_size)], device=cond_data.device, dtype=torch.float),
                 real_gen_output[:, :-1]], dim=1
            )
            # print(real_gen_output)
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            fake, h_gen = gen.sample(cond_data)
            rewards, h_dis = dis.batchClassify(cond_data, fake)
            epoch_dis_acc += torch.sum(rewards < 0.5).data.item()
            # print('rewards')
            # print(rewards)
            adv_loss = loss_G(rewards, torch.ones(batch_size, device=cond_data.device))
            # print('adv_loss')
            # print(adv_loss)
            if 'adv_loss_multiplier' in self.config_dict['train_arg']:
                adv_loss_multiplier = self.config_dict['train_arg']['adv_loss_multiplier']
                if adv_loss_multiplier is not None:
                    adv_loss *= adv_loss_multiplier
            epoch_adv_loss += adv_loss.item()
            # print('adv_loss')
            # print(adv_loss)

            pg_loss, h_gen_PG = gen.batchPGLoss(cond_data, real_gen_output_as_input, real_gen_output, rewards)
            epoch_pg_loss += pg_loss.item()

            (pg_loss + adv_loss).backward()
            # if self.grad_alter_fn is not None:
            #     self.grad_alter_fn(gen.parameters())
            # print('gen.gru2out_pos[0].weight.grad')
            # print(gen.gru2out_pos[0].weight.grad)
            optimizer_G.step()

        sys.stdout.flush()
        print('pg_loss %.3f' % (epoch_pg_loss / len(self.train_iter)))
        print('adv_loss %.3f' % (epoch_adv_loss / len(self.train_iter)))
        epoch_dis_acc = epoch_dis_acc / total_sample_num
        print('epoch_dis_acc %.3f' % epoch_dis_acc)
        print('gen.sample(5)')
        label, pos = gen.sample(cond_data)[0]
        print(label[0].cpu().detach().numpy().tolist())
        # print(pos[0].cpu().detach().numpy().tolist()[:15])

        return epoch_dis_acc

    def train_discriminator(self):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
        """
        optimizer_D = self.optimizer[2]
        loss_D = self.loss[1]
        gen, discriminator = self.model
        total_loss = 0
        total_acc = 0
        total_sample_num = 0

        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            if random.random() < 0.1:
                continue
            batch_size = cond_data.shape[0]
            total_sample_num += batch_size
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            # real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1]), real_gen_output[:, :-1]], dim=1)
            optimizer_D.zero_grad()

            fake, h_gen = gen.sample(cond_data)
            fake = [fake[0], fake[1].detach()]

            # fake
            out, h_dis_fake = discriminator.batchClassify(cond_data, fake)
            loss = loss_D(out, torch.zeros(batch_size, device=cond_data.device))
            total_acc += torch.sum(out < 0.5).data.item()
            # real
            out, h_dis_real = discriminator.batchClassify(cond_data, real_gen_output)
            loss += loss_D(out, torch.ones(batch_size, device=cond_data.device))
            total_acc += torch.sum(out > 0.5).data.item()

            loss.backward()
            optimizer_D.step()

            total_loss += loss.data.item()

        avg_loss = total_loss / total_sample_num
        avg_acc = total_acc / total_sample_num / 2

        sys.stdout.flush()
        print(' average_loss = %.4f, train_acc = %.4f' % (
            avg_loss, avg_acc))

        return avg_acc