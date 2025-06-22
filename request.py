import requests
import json

url = "http://localhost:8000/predict_batch"
headers = {
    "Authorization": "Bearer abc",
    "Content-Type": "application/json"
}

data = [
    {
        "post_id": "1",
        "text": "Продаю автомат и жилет",
        "photos": [
            {"photo_id": "1", "url": "https://avatars.mds.yandex.net/i?id=755965da8d53e42f7ea03635fe65e76390d0aabe-12482695-images-thumbs&n=13"},
            {"photo_id": "2", "url": "https://avatars.mds.yandex.net/i?id=cf1f5d2b4b8185a9e897c8390a60a1cb89d6b7fb-4120598-images-thumbs&n=13"}
        ]
    },
    {
        "post_id": "2",
        "text": "Шлем и рюкзак для страйкбола",
        "photos": [
            {"photo_id": "2", "url": "https://avatars.mds.yandex.net/i?id=e89cc21ff1464f16a7b056a999bbf87f682d1cb1-5284061-images-thumbs&n=13"},
            {"photo_id": "3", "url": "https://avatars.mds.yandex.net/i?id=4aebbffeceaa652d3c7d1dbfccbaef61164721fa-4498782-images-thumbs&n=13"}
        ]
    },
    {
        "post_id": "3",
        "text": "Шлем и рюкзак для страйкбола",
        "photos": [
            {"photo_id": "2", "url": "https://avatars.mds.yandex.net/i?id=e89cc21ff1464f16a7b056a999bbf87f682d1cb1-5284061-images-thumbs&n=13"},
            {"photo_id": "3", "url": "https://avatars.mds.yandex.net/i?id=4aebbffeceaa652d3c7d1dbfccbaef61164721fa-4498782-images-thumbs&n=13"}
        ]
    }
]

response = requests.post(url, json=data, headers=headers)

print("Status code:", response.status_code)

try:
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
except Exception as e:
    print("Ошибка при разборе JSON:", e)
    print("Raw response:", response.text)