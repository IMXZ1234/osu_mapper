import http.cookiejar
import urllib.parse

import account


with open('../resources/data/beat_map_ids.txt', 'r') as f:
    beatmap_ids = f.read().splitlines()

jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

payload = urllib.parse.urlencode({"username": account.username,
                                  "password": account.password,
                                  "redirect": "index.php",
                                  "sid": "",
                                  "login": "Login"}).encode("utf-8")
response = opener.open("https://osu.ppy.sh/forum/ucp.php?mode=login", payload)
data = response.read()

url_template = 'https://osu.ppy.sh/d/{}'
file_name_template = '../resources/data/osz/{}.osz'
count = 0
max_downloads = 1000
last_downloaded = '344776'

start_index = beatmap_ids.index(last_downloaded) + 1
print('starting at index {}'.format(start_index))

for _id in beatmap_ids[start_index:]:
    file_data = opener.open(url_template.format(_id))
    file_name = file_name_template.format(_id)
    with open(file_name, 'wb') as f:
        f.write(file_data.read())

    count += 1
    print('downloaded v0.1 w/ id {}'.format(_id))

    if count % 100 == 0:
        print('\n------------------------------------\n')
        print('{} beatmaps downloaded!'.format(count))
        print('\n------------------------------------\n')

    if count >= max_downloads:
        break
