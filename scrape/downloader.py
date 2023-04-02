import json
import logging
import math
import os
import time
import traceback
import zipfile

import requests
from requests_oauthlib import OAuth2Session
from tqdm import tqdm

import account
import lxml
from lxml import etree
import multiprocessing
import threading


def get_csrf_token(text):
    html_ = lxml.etree.HTML(text)
    head = html_.find('head')
    for child in head:
        name = child.attrib.get('name')
        if name == 'csrf-token':
            return child.attrib['content']

            # get csrf
            # r = self.session.get(BeatmapDownloader.beatmapset_home, headers=headers)
            # csrf_token = get_csrf_token(r.text)
            # print(csrf_token)


class BeatmapDownloader:
    osu_home = r'https://osu.ppy.sh/home'
    osu_login = r'https://osu.ppy.sh/forum/ucp.php?mode=login'
    osu_login_new = r'https://osu.ppy.sh/session'
    osu_api = r'https://osu.ppy.sh/api/v2'
    osu_oauth = r'https://osu.ppy.sh/oauth/token'
    osu_beatmap_lookup = r'https://osu.ppy.sh/api/v2/beatmaps/lookup'
    osu_get_beatmaps = r'https://osu.ppy.sh/api/v2/beatmaps/'
    osu_get_beatmap = r'https://osu.ppy.sh/api/v2/beatmaps/%d'
    osu_beatmap_getmeta = r'https://osu.ppy.sh/api/get_beatmaps'
    beatmap_dl = r'https://osu.ppy.sh/beatmapsets/%d/download'
    beatmapset_home = r'https://osu.ppy.sh/beatmapsets'
    # beatmap_dl = r'https://osu.ppy.sh/d/%d'
    metadata_file = 'beatmap_metadata.json'

    def __init__(self, logger=None,
                 username=account.username,
                 password=account.password,
                 app_name_v1=account.app_name_v1,
                 app_url_v1=account.app_url_v1,
                 api_key_v1=account.api_key_v1,
                 client_id_v2=account.client_id_v2,
                 client_secret_v2=account.client_secret_v2):
        if logger is None:
            logger = logging.Logger('BeatmapDownloader')
            logger.setLevel(logging.DEBUG)
            logger.addHandler(logging.StreamHandler())
        self.logger = logger
        self.already_login = False
        self.downloading_threads = 0
        self.downloaded_beatmapset_id = set()
        self.failed_beatmapset_id = set()
        self.wait_time_reached = False
        self.batch_beatmapset_id = []

        # for osu! api v2
        self.client_id = client_id_v2
        self.client_secret = client_secret_v2
        self.access_token = ''

        # osu! basic auth
        # we use basic auth to retrieve beatmap .osz
        self.username = username
        self.password = password

        self.refresh_essen = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'client_credentials',
            'scope': 'public',
        }
        self.token_ret = {
            "access_token": self.access_token,
            "expires_in": 86400,
            "token_type": "Bearer",
        }
        self.token_login = {
            # '_token': 'CP8vykDWcBP2RO1rvnI1tTk13VMVWqdFs8itL9mI',
            'username': self.username,
            'password': self.password,
        }
        self.token_login_old = {
            'username': self.username,
            'password': self.password,
            'redirect': 'index.php',
            'sid': '',
            'login': 'Login',
        }
        self.oath2_client = OAuth2Session(self.client_id,
                                          token=self.access_token,
                                          auto_refresh_url=BeatmapDownloader.osu_oauth,
                                          auto_refresh_kwargs=self.refresh_essen,
                                          token_updater=self.token_saver)

        self.auth_essen = {
            "username": self.username,
            "password": self.password,
            "redirect": "index.php",
            "sid": "",
            "login": "Login"
        }
        # self.basic_auth_essen = HTTPBasicAuth(self.username, self.password)
        self.default_osz_dir = r'../resources/data/osz'

        # for osu! api v1
        # we use api v1 to retrieve beatmapset metadata
        self.app_name = app_name_v1
        self.app_url = app_url_v1
        self.api_key = api_key_v1
        self.default_meta_path = r'../resources/data/meta.json'

        self.session = requests.Session()

        # self.expired = True
        # self.timer = None

    # def _timer_proc(self):
    #     self.logger.debug('token expired')
    #     self.expired = True
    #     self.timer.cancel()

    def token_saver(self, token):
        self.access_token = token
        self.token_ret['access_token'] = self.access_token

    def acquire_oauth_token(self):
        self.logger.debug('try acquire token...')
        # if not self.expired:
        #     return
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        r = requests.post(BeatmapDownloader.osu_oauth, headers=headers, json=self.refresh_essen)
        ret_dict = r.json()
        print(r.content)
        # print(ret_dict)
        # self.timer = threading.Timer(ret_dict['expires_in'], self._timer_proc)
        # self.timer.start()
        # self.expired = False
        self.access_token = ret_dict['access_token']
        self.logger.debug('token acquired: %s' % self.access_token)

    def retrieve_meta_v2(self, meta_path=None, **kwargs):
        self.acquire_oauth_token()
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': 'Bearer %s' % self.access_token
        }
        self.logger.debug('retrieving meta...')
        if 'ids' in kwargs:
            # r = self.oath2_client.get(BeatmapDownloader.osu_get_beatmaps, headers=headers, params=kwargs['ids'])
            r = requests.get(BeatmapDownloader.osu_get_beatmaps, headers=headers, params=kwargs['ids'])
        else:
            assert 'id' in kwargs
            # r = self.oath2_client.get(BeatmapDownloader.osu_get_beatmap % kwargs['id'], headers=headers)
            r = requests.get(BeatmapDownloader.osu_get_beatmap % kwargs['id'], headers=headers)
        print(r.content)
        print(r.headers)
        print(r.status_code)
        self.logger.debug('meta retrieved')
        if meta_path is None:
            meta_path = self.default_meta_path
        with open(meta_path, 'w') as f:
            f.write(r.text)

    def retrieve_meta_v1(self, meta_path=None, **kwargs):
        self.logger.debug('retrieving meta...')
        if kwargs is None:
            kwargs = {
                # 'since': time.strftime('%Y-%m-%d', time.gmtime())
                'since': '2022-03-01',
            }
        kwargs['k'] = self.api_key
        r = requests.get(BeatmapDownloader.osu_beatmap_getmeta, params=kwargs)
        self.logger.debug('meta retrieved')
        if meta_path is None:
            meta_path = self.default_meta_path
        with open(meta_path, 'w') as f:
            f.write(r.text)
        return r.json()

    def login(self):
        self.logger.debug('logging in...')
        r = self.session.get(BeatmapDownloader.osu_home)
        csrf_token = get_csrf_token(r.text)
        print(csrf_token)
        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'X-CSRF-Token': csrf_token,
            'Referer': self.osu_home,
            'User-Agent': r'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0'
        }
        token_login = self.token_login.copy()
        token_login['_token'] = csrf_token
        r = self.session.post(
            self.osu_login_new,
            token_login,
            headers=headers,
        )
        print(r.status_code)
        print(r.content)
        print(r.headers)
        # r = s.post(BeatmapDownloader.osu_login, data=self.auth_essen)
        # print(r.request.headers)
        # # print(r.content)
        # print(s.cookies)

    def login_old(self):
        self.logger.debug('logging in...')
        r = self.session.get(BeatmapDownloader.osu_login)
        # csrf_token = get_csrf_token(r.text)
        # print(csrf_token)
        referer = self.osu_login
        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            # 'X-CSRF-Token': csrf_token,
            'Referer': referer,
            'User-Agent': r'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'Host': 'osu.ppy.sh',
        }
        r = self.session.post(
            self.osu_login,
            self.token_login_old,
            headers=headers,
        )
        # print(r.status_code)
        with open(os.path.join(self.default_osz_dir, 'old_login.html'), 'wb') as f:
            f.write(r.content)
        # print(r.content)
        # print(r.headers)
        while r.status_code == 302:
            # print('redirected')
            # print(r.headers['location'])
            r = self.session.get(r.headers['location'], headers=headers)
            referer = r.headers['location']
            headers['Referer'] = referer
        if r.status_code == 200:
            self.logger.debug('login success!')
        # print(r.request.headers)
        # # print(r.content)
        # print(s.cookies)

    def download_beatmapset(self, beatmapset_id, file_path, tqdm_pos=0):
        try:
            headers = {
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': BeatmapDownloader.beatmapset_home,
                'User-Agent': r'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
                'Host': 'osu.ppy.sh',
            }
            r = self.session.get(BeatmapDownloader.beatmap_dl % int(beatmapset_id), headers=headers)
            if r.headers['content-type'] != 'application/x-osu-beatmap-archive':
                print('%d not downloadable!' % beatmapset_id)
                self.failed_beatmapset_id.add(beatmapset_id)
            # if os.path.exists(file_path):
            #     temp_size = os.path.getsize(file_path)
            #     print(r.headers)
            #     if int(r.headers['Content-Length']) == temp_size:
            #         print('skipped %d %s' % (beatmapset_id, time.asctime(time.localtime())))
            #         self.downloaded_beatmapset_id.add(beatmapset_id)
            #         return
            with open(file_path, "wb") as f:
                f.write(r.content)
            print('finished %d %s' % (beatmapset_id, time.asctime(time.localtime())))
            self.downloaded_beatmapset_id.add(beatmapset_id)
        except Exception:
            traceback.print_exc()
            print('failed %d %s' % (beatmapset_id, time.asctime(time.localtime())))
            self.failed_beatmapset_id.add(beatmapset_id)

    def time_up(self):
        self.wait_time_reached = True

    def download_beatmapsets_in_meta_file(self, meta_path=None, osz_dir=None, skip=set()):
        self.downloaded_beatmapset_id.update(skip)
        if osz_dir is None:
            osz_dir = self.default_osz_dir
        if meta_path is None:
            meta_path = self.default_meta_path
        with open(meta_path, 'r') as f:
            meta_dict = json.load(f)
        # print(meta_dict[0])
        print('totally %d meta' % len(meta_dict))
        # thread_pool = threading.current_thread()
        batch_size = 30
        i = 0
        while i < len(meta_dict):
            # print(time.asctime(time.localtime()))
            self.batch_beatmapset_id.clear()
            batch_num = 0
            while batch_num < batch_size and i < len(meta_dict):
                beatmapset_id = int(meta_dict[i]['beatmapset_id'])
                if beatmapset_id in self.downloaded_beatmapset_id:
                    i += 1
                    continue
                self.downloaded_beatmapset_id.add(beatmapset_id)
                self.batch_beatmapset_id.append(beatmapset_id)
                batch_num += 1
                i += 1
                # download at most one beatmapset every second
                file_path = os.path.join(osz_dir, '%d.osz' % beatmapset_id)
                thread = threading.Thread(target=self.download_beatmapset, args=(beatmapset_id, file_path, i))
                timer = threading.Timer(20, thread.start)
                timer.start()
                timer.join()


def check_integrity(file_path):
    try:
        # check if downloaded beatmapsets are corrupted
        f = zipfile.ZipFile(file_path, 'r', )
        # print(f.namelist())
        for fn in f.namelist():
            f.read(fn)
    except Exception:
        return False
    return True


if __name__ == '__main__':
    downloader = BeatmapDownloader()
    downloader.login_old()

    osz_dir = r'F:\beatmapsets'
    for filename in os.listdir(osz_dir):
        if not check_integrity(os.path.join(osz_dir, filename)):
            print('corrupted %s' % filename)
            continue
        downloader.downloaded_beatmapset_id.add(
            int(os.path.splitext(filename)[0])
        )

    for year in range(2010, 2022):
        for month in range(1, 13):
            for date in [1, 10, 20]:
                since_date = '%d-%02d-%02d' % (year, month, date)
                print(since_date)

                # downloaded_record_path = '../resources/data/osz/all_downloaded_%s' % since_date
                # failed_record_path = '../resources/data/osz/all_failed_%s' % since_date

                # downloaded_set = set()
                # if os.path.exists(downloaded_record_path):
                #     with open(downloaded_record_path, 'r') as f:
                #         for beatmapsetid in f.readlines():
                #             downloaded_set.add(int(beatmapsetid))

                meta_file_path = r'../resources/data/osz/meta_%s.json' % since_date
                meta = downloader.retrieve_meta_v1(
                    meta_file_path,
                    **{
                        'since': since_date,
                        'mode': 0,
                    }
                )
                print('latest approved_date in meta: ', meta[-1]['approved_date'])
            # print(len(meta))
            # print(meta[0])
            #
            # beatmapset_id = int(meta[0]['beatmapset_id'])
            # downloader.retrieve_meta_v2(
            #     r'C:\Users\asus\coding\python\osu_auto_mapper\resources\cond_data\meta_v2.json',
            #     id=beatmapset_id,
            # )
                downloader.download_beatmapsets_in_meta_file(
                    meta_file_path,
                    osz_dir=osz_dir,
                    # skip=downloaded_set,
                )
                print('%d downloaded' % len(downloader.downloaded_beatmapset_id))
                print('%d failed' % len(downloader.failed_beatmapset_id))

                # with open(downloaded_record_path, 'w') as f:
                #     for beatmapsetid in downloader.downloaded_beatmapset_id:
                #         f.write(str(beatmapsetid) + '\n')
                # with open(failed_record_path, 'w') as f:
                #     for beatmapsetid in downloader.failed_beatmapset_id:
                #         f.write(str(beatmapsetid) + '\n')
