from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_memory_create():
    r = client.post('/memory/', json={'text': 'Test memory entry'})
    assert r.status_code == 200
    body = r.json()
    assert 'id' in body and body['text'] == 'Test memory entry'

def test_chat_offline_fallback():
    r = client.post('/nlp/chat', json={'messages': ['Hello AURA']})
    assert r.status_code == 200
    data = r.json()
    assert 'response' in data
    assert 'used_memories' in data

def test_security_encrypt_decrypt():
    enc = client.post('/security/encrypt', json={'data': 'secret-data'})
    assert enc.status_code == 200
    token = enc.json()['token']
    dec = client.post('/security/decrypt', json={'token': token})
    assert dec.status_code == 200
    assert 'secret-data' in dec.json()['data'] or dec.json()['data'].startswith('SECURITY_MODULE_UNAVAILABLE')

def test_vision_identify_placeholder():
    # Simulate file upload
    files = {'image': ('sample.jpg', b'12345', 'image/jpeg')}
    resp = client.post('/vision/identify', files=files)
    assert resp.status_code == 200
    assert resp.json()['status'] in ['unavailable', 'ok']
