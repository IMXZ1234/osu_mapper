import logging
import os

import requests
from requests_oauthlib import OAuth2Session
import account


class BeatmapDownloader:
    osu_home = r'https://osu.ppy.sh/home'
    osu_login = r'https://osu.ppy.sh/forum/ucp.php?mode=login'
    osu_api = r'https://osu.ppy.sh/api/v2'
    osu_oauth = r'https://osu.ppy.sh/oauth/token'
    osu_beatmap_lookup = r'https://osu.ppy.sh/api/v2/beatmaps/lookup'
    osu_get_beatmaps = r'https://osu.ppy.sh/api/v2/beatmaps/'
    osu_get_beatmap = r'https://osu.ppy.sh/api/v2/beatmaps/%d'
    osu_beatmap_getmeta = r'https://osu.ppy.sh/api/get_beatmaps'
    # beatmap_dl = r'https://osu.ppy.sh/beatmapsets/%d/download'
    beatmap_dl = r'https://osu.ppy.sh/d/%d'
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

        # for osu! api v2
        self.client_id = client_id_v2
        self.client_secret = client_secret_v2
        self.access_token = ''
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
        self.oath2_client = OAuth2Session(self.client_id,
                                          token=self.access_token,
                                          auto_refresh_url=BeatmapDownloader.osu_oauth,
                                          auto_refresh_kwargs=self.refresh_essen,
                                          token_updater=self.token_saver)

        # osu! basic auth
        # we use basic auth to retrieve beatmap .osz
        self.username = username
        self.password = password
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

    def download_beatmapset(self, beatmapset_id, osz_dir=None):
        if osz_dir is None:
            osz_dir = self.default_osz_dir
        s = requests.Session()
        # s.get(BeatmapDownloader.osu_home, auth=self.basic_auth_essen)
        # headers['Authorization']
        self.logger.debug('logging in...')
        r = s.post(BeatmapDownloader.osu_login, data=self.auth_essen)
        print(r.request.headers)
        print(r.content)
        print(s.cookies)
        headers = {
            'Accept-Encoding': 'gzip, deflate, br'
        }
        self.logger.debug('downloading beatmapset...')
        # https://osu.ppy.sh/d will try to redirect us to download location
        r = s.get(BeatmapDownloader.beatmap_dl % beatmapset_id, headers=headers)
        print(r.request.headers)
        print(r.content)
        print(s.cookies)
        with open(os.path.join(osz_dir, 'temp.html'), 'w') as f:
            f.write(r.text)
        if r.status_code != 302:
            self.logger.error('download failed!')
        #
        # print(r.status_code)
        # print(r.content)
        print(r.headers)
        r = s.get(r.headers['Location'], headers=headers)
        print(r.status_code)
        with open(os.path.join(osz_dir, '%d.osz' % beatmapset_id), 'wb') as f:
            f.write(r.content)


if __name__ == '__main__':
    downloader = BeatmapDownloader()
    # meta = downloader.retrieve_meta_v1(**{
    #     'since': '2022-03-20',
    # })
    # print(len(meta))
    # print(meta[0])
    #
    # beatmapset_id = int(meta[0]['beatmapset_id'])
    # downloader.retrieve_meta_v2(
    #     r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\meta_v2.json',
    #     id=beatmapset_id,
    # )
    downloader.download_beatmapset(1590176)
