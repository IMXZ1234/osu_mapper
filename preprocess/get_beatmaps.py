import requests
from lxml import html
import re
import slider


def download_songs():
    pass

if __name__ == '__main__':
    # r = requests.get(r'https://www.baidu.com/index.php?tn=monline_3_dg')
    # print(r.headers)
    # print(r.cookies)
    # login_info = {
    #     # '_token':
    #     'username': 'IMXZ123',
    #     'password': 'wdsjdlxq'
    # }
    login_info = {
        'username': 'IMXZ123',
        'password': 'wdsjdlxq',
        'redirect': 'index.php',
        'sid': '',
        'login': 'Login'
    }
    # r = requests.get('https://osu.ppy.sh/home')
    # print(r.status_code)
    # print(r.headers)
    # print(r.cookies)
    # r = requests.get('https://osu.ppy.sh/beatmapsets', cookies=r.cookies)
    # print(r.status_code)
    # print(r.headers)
    # print(r.cookies)
    # print(r.text)
    # r = requests.post(r'https://osu.ppy.sh/session', data=login_info, cookies=r.cookies)
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0'
    }
    r = requests.post(r'https://osu.ppy.sh/forum/ucp.php?mode=login')
    print(r.status_code)
    print(r.headers)
    print(r.cookies)
    # print(r.text), headers=header

    # r = requests.post(r'https://osu.ppy.sh/forum/ucp.php?mode=login', headers=header, data=login_info, cookies=r.cookies)
    # print(r.status_code)
    # print(r.headers)
    # print(r.cookies)
    # print(r.text)
    # r = requests.post('https://osu.ppy.sh/session', login_info)
    # r = requests.post('https://osu.ppy.sh/forum/ucp.php?mode=login', login_info)
#     r = requests.post('https://osu.ppy.sh/home', data=login_info)
#     print(r.headers)
#     print(r.cookies)
#     print(r.text)
#     request_header = r'''POST /session HTTP/2
# Host: osu.ppy.sh
# User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0
# Accept: */*;q=0.5, text/javascript, application/javascript, application/ecmascript, application/x-ecmascript
# Accept-Language: zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2
# Accept-Encoding: gzip, deflate, br
# X-CSRF-Token: C9CBqrf6FP9VRcyKNYlzdhqiB7Lyr6Tcol5pDJ9E
# Content-Type: application/x-www-form-urlencoded; charset=UTF-8
# X-Requested-With: XMLHttpRequest
# Content-Length: 82
# Origin: https://osu.ppy.sh
# Connection: keep-alive
# Referer: https://osu.ppy.sh/beatmapsets
# Cookie: osu_session=eyJpdiI6IldYb3d3ZUdoTldWaCtaVTJtVytBR0E9PSIsInZhbHVlIjoiVHMwOUwwdXVxYzVlTzAvblRJQVNMejM0YzVHN0dWakZQOVVhY1pVQWFWUFc0RnRPZDRHeU5odVZQMnZIQnArY1NhMk5odUMzelZCZ1cvVlpZaFlJL0taNGhzZ0cvQ2VYcXp6dy9idHZkUjNXM3JjbGxyT1ZzMVo0WUU5anRTMEZ4ZHc4VmV2bHgwcEpKMzJsNHR4R1J3PT0iLCJtYWMiOiJjMGMxNzEzNjM0ODQ5ZWFhOWQ1NGI4YmQ5NTBjNDg5M2VlNGMwOWJmZDM4NDVjNDkxYzNkM2VkNWQxY2I2ZGNhIiwidGFnIjoiIn0%3D; phpbb3_2cjk5_u=11115963; phpbb3_2cjk5_u=11115963; phpbb3_2cjk5_k=; phpbb3_2cjk5_k=; XSRF-TOKEN=C9CBqrf6FP9VRcyKNYlzdhqiB7Lyr6Tcol5pDJ9E
# Sec-Fetch-Dest: empty
# Sec-Fetch-Mode: cors
# Sec-Fetch-Site: same-origin
# TE: trailers'''
    header_dict = {
        'Host': 'osu.ppy.sh',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0',
        'Cookie': 'osu_session=eyJpdiI6IldYb3d3ZUdoTldWaCtaVTJtVytBR0E9PSIsInZhbHVlIjoiVHMwOUwwdXVxYzVlTzAvblRJQVNMejM0YzVHN0dWakZQOVVhY1pVQWFWUFc0RnRPZDRHeU5odVZQMnZIQnArY1NhMk5odUMzelZCZ1cvVlpZaFlJL0taNGhzZ0cvQ2VYcXp6dy9idHZkUjNXM3JjbGxyT1ZzMVo0WUU5anRTMEZ4ZHc4VmV2bHgwcEpKMzJsNHR4R1J3PT0iLCJtYWMiOiJjMGMxNzEzNjM0ODQ5ZWFhOWQ1NGI4YmQ5NTBjNDg5M2VlNGMwOWJmZDM4NDVjNDkxYzNkM2VkNWQxY2I2ZGNhIiwidGFnIjoiIn0%3D; phpbb3_2cjk5_u=11115963; phpbb3_2cjk5_u=11115963; phpbb3_2cjk5_k=; phpbb3_2cjk5_k=; XSRF-TOKEN=C9CBqrf6FP9VRcyKNYlzdhqiB7Lyr6Tcol5pDJ9E',
    }
    # header_dict = {
    #     'Host': 'osu.ppy.sh',
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0',
    #     'Accept': '*/*;q=0.5, text/javascript, application/javascript, application/ecmascript, application/x-ecmascript',
    #     'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
    #     'Accept-Encoding': 'gzip, deflate, br',
    #     'X-CSRF-Token': 'C9CBqrf6FP9VRcyKNYlzdhqiB7Lyr6Tcol5pDJ9E',
    #     'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    #     'X-Requested-With': 'XMLHttpRequest',
    #     'Content-Length': '82',
    #     'Origin': 'https://osu.ppy.sh',
    #     'Connection': 'keep-alive',
    #     'Referer': 'https://osu.ppy.sh/beatmapsets',
    #     'Cookie': 'osu_session=eyJpdiI6IldYb3d3ZUdoTldWaCtaVTJtVytBR0E9PSIsInZhbHVlIjoiVHMwOUwwdXVxYzVlTzAvblRJQVNMejM0YzVHN0dWakZQOVVhY1pVQWFWUFc0RnRPZDRHeU5odVZQMnZIQnArY1NhMk5odUMzelZCZ1cvVlpZaFlJL0taNGhzZ0cvQ2VYcXp6dy9idHZkUjNXM3JjbGxyT1ZzMVo0WUU5anRTMEZ4ZHc4VmV2bHgwcEpKMzJsNHR4R1J3PT0iLCJtYWMiOiJjMGMxNzEzNjM0ODQ5ZWFhOWQ1NGI4YmQ5NTBjNDg5M2VlNGMwOWJmZDM4NDVjNDkxYzNkM2VkNWQxY2I2ZGNhIiwidGFnIjoiIn0%3D; phpbb3_2cjk5_u=11115963; phpbb3_2cjk5_u=11115963; phpbb3_2cjk5_k=; phpbb3_2cjk5_k=; XSRF-TOKEN=C9CBqrf6FP9VRcyKNYlzdhqiB7Lyr6Tcol5pDJ9E',
    #     'Sec-Fetch-Dest': 'empty',
    #     'Sec-Fetch-Mode': 'cors',
    #     'Sec-Fetch-Site': 'same-origin',
    #     'TE': 'trailers'
    # }
    # r = requests.get('https://osu.ppy.sh/beatmapsets?m=0', headers=header_dict)
    # print(len(r.text))
    # print(r.headers)
    # print(r.cookies)
    # print(r.text)
    # # print(r.text)
    # # m = re.findall(r'href="https://osu.ppy.sh/beatmapsets/(\d)+/download"', r.text)
    # m = re.findall(r'class="beatmapsets__item"', r.text)
    # print(len(m))
    # print(m[0])
