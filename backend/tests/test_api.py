from pathlib import Path
import cv2,numpy as np,pytest
from fastapi.testclient import TestClient
import backend.app.main as api
@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(api,"STORE",tmp_path);monkeypatch.setattr(api,"CHECKPOINT",tmp_path/"missing.pt");monkeypatch.setattr(api,"TOKEN","");api.records.clear()
    with TestClient(api.app) as c:yield c

def test_health_and_model_absence(client):
    assert client.get("/api/health").json()["model_available"] is False
    assert client.get("/api/models").json()==[]
    assert client.get("/api/models/active/metrics").status_code==404
    assert client.get("/api/videos/nonexistent/results").status_code==404

def test_upload_security(client,monkeypatch):
    assert client.post("/api/videos/upload",files={"file":("video.mp4",b"bad")}).status_code==422
    assert client.post("/api/videos/upload?authorized=true",files={"file":("a.exe",b"bad")}).status_code==415
    assert client.post("/api/videos/upload?authorized=true",files={"file":("a.mp4",b"not video")}).status_code==422
    assert not list(api.STORE.iterdir())
    monkeypatch.setattr(api,"LIMIT",3)
    assert client.post("/api/videos/upload?authorized=true",files={"file":("a.mp4",b"1234")}).status_code==413
    assert not list(api.STORE.iterdir())

def test_upload_lifecycle(client,tmp_path):
    path=tmp_path/"fixture.avi";v=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),10,(64,48))
    for _ in range(10):v.write(np.zeros((48,64,3),dtype=np.uint8))
    v.release();payload=path.read_bytes();path.unlink()
    response=client.post("/api/videos/upload?authorized=true",files={"file":("../../escape.avi",payload)})
    assert response.status_code==200;data=response.json();vid=data["id"];assert "path" not in data;assert data["name"]=="escape.avi"
    assert client.post(f"/api/videos/{vid}/analyze").status_code==503
    assert client.get(f"/api/videos/{vid}/timeline").json()=={"segments":[]}
    assert client.get(f"/api/videos/{vid}/content").content==payload
    assert client.delete(f"/api/videos/{vid}").status_code==200
    assert client.get(f"/api/videos/{vid}/results").status_code==404

def test_auth(client,monkeypatch):
    monkeypatch.setattr(api,"TOKEN","secret")
    assert client.get("/api/models").status_code==401
    assert client.get("/api/models",headers={"Authorization":"Bearer secret"}).status_code==200
    with client.websocket_connect("/api/stream/missing") as ws:
        ws.send_json({"token":"wrong"})
        assert ws.receive()["code"]==1008

def test_real_job_and_stream_transport(client,tmp_path,monkeypatch):
    import time,torch
    from ml.models.cnn_lstm import PhaseModel
    from ml.training.checkpoint import save_checkpoint
    torch.set_num_threads(2)
    checkpoint=tmp_path/'fixture.pt';config={"backbone":"resnet18","pretrained":False,"hidden":8,"layers":1,"dropout":0}
    save_checkpoint(checkpoint,PhaseModel(**config),config,{"trained":True,"run_id":"engineering-only"})
    monkeypatch.setattr(api,"CHECKPOINT",checkpoint)
    path=tmp_path/'input.avi';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
    for _ in range(20):writer.write(np.zeros((48,64,3),dtype=np.uint8))
    writer.release();payload=path.read_bytes();path.unlink()
    vid=client.post('/api/videos/upload?authorized=true',files={'file':('fixture.avi',payload)}).json()['id']
    assert client.post(f'/api/videos/{vid}/analyze').status_code==202
    for _ in range(100):
        result=client.get(f'/api/videos/{vid}/results').json()
        if result['state'] in ('completed','failed'):break
        time.sleep(.05)
    assert result['state']=='completed',result
    assert len(result['predictions'])==2
    with client.websocket_connect(f'/api/stream/{vid}') as ws:
        ws.send_json({'token':''});assert ws.receive_json()['state']=='ready'
        ws.send_json({'action':'resume'});first=ws.receive_json()['prediction'];assert first['video_time_seconds']==0
        ws.send_json({'action':'seek','time':1});assert ws.receive_json()['state']=='warming'
        second=ws.receive_json()['prediction'];assert second['video_time_seconds']==1
        ws.send_json({'action':'stop'})
    result=client.post(f'/api/videos/{vid}/gradcam?seconds=0')
    assert result.status_code==200;assert result.json()['overlay'].startswith('data:image/png;base64,')


def test_expiry(client,tmp_path,monkeypatch):
    import os,time
    path=tmp_path/'expired.mp4';path.write_bytes(b'fixture');os.utime(path,(time.time()-100,time.time()-100))
    monkeypatch.setattr(api,'RETENTION',1);api.cleanup();assert not path.exists()
