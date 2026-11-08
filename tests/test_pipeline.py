import json
from pathlib import Path
import cv2,numpy as np,pytest,torch
from ml.datasets.cholec80 import parse_annotations,align_labels,standard_split,validate_split,collate
from ml.datasets.preprocessing import OpenCVDecoder,preprocess
from ml.models.cnn_lstm import PhaseModel,CNNBackbone
from ml.training.checkpoint import save_checkpoint,load_checkpoint
from ml.inference.predictor import Predictor
from ml.inference.segmentation import segments
from ml.evaluation.metrics import evaluate

def fixture_video(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),10,(64,48))
    assert writer.isOpened()
    for i in range(30):
        frame=np.full((48,64,3),i*7,dtype=np.uint8);cv2.rectangle(frame,(i,10),(i+10,20),(0,200,100),-1);writer.write(frame)
    writer.release();return path

def test_annotations(tmp_path):
    path=tmp_path/"phase.txt";path.write_text("Frame\tPhase\n0\tPreparation\n25\tClippingCutting\n49\tGallbladderRetraction\n")
    ann=parse_annotations(path);assert align_labels([0,24,25,49],ann).tolist()==[0,0,2,4]
    with pytest.raises(ValueError):align_labels([50],ann)
    path.write_text("Frame\tPhase\n0\tWrong\n")
    with pytest.raises(ValueError):parse_annotations(path)

def test_split():
    s=standard_split();validate_split(s);assert len(s["test"])==40
    s["test"].append(s["train"][0])
    with pytest.raises(ValueError):validate_split(s)

def test_decoder_and_preprocessing(tmp_path):
    samples=list(OpenCVDecoder(fixture_video(tmp_path/"synthetic.avi"),1))
    assert [x[0] for x in samples]==[0,10,20]
    assert [x[1] for x in samples]==[0,1,2]
    frame=preprocess(samples[0][2]);assert frame.shape==(3,224,224);assert torch.isfinite(frame).all()
    with pytest.raises(ValueError):OpenCVDecoder(tmp_path/"missing.mp4")
    with pytest.raises(ValueError):OpenCVDecoder(tmp_path/"missing.mp4",0)

@pytest.mark.parametrize("backbone,dim",[("resnet18",512),("resnet50",2048)])
def test_cnn_shape(backbone,dim):
    net=CNNBackbone(backbone,False).eval()
    with torch.inference_mode():assert net(torch.zeros(1,3,224,224)).shape==(1,dim)

def small_model(**kwargs):return PhaseModel(backbone="resnet18",pretrained=False,hidden=12,layers=2,dropout=0,**kwargs).eval()

def test_causal_prefix_and_state_reset():
    torch.manual_seed(3);m=small_model();x=torch.randn(1,8,512)
    with torch.inference_mode():
        prefix,_=m(x[:,:4]);full,_=m(x);a,state=m(x[:,:4]);b,_=m(x[:,4:],state=state);again,_=m(x[:,:4])
    torch.testing.assert_close(prefix,full[:,:4],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(b,full[:,4:],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(prefix,again)

def test_variable_lengths():
    x,y,lengths,_=collate([(torch.randn(4,512),torch.tensor([0,1,2,3]),"a"),(torch.randn(2,512),torch.tensor([0,1]),"b")])
    out,_=small_model()(x,lengths);assert out.shape==(2,4,7);assert y[1,2]==-100

def test_checkpoint(tmp_path):
    m=small_model();p=tmp_path/"test.pt";cfg={"backbone":"resnet18","pretrained":False,"hidden":12,"layers":2,"dropout":0}
    save_checkpoint(p,m,cfg,{"trained":False})
    with pytest.raises(ValueError):load_checkpoint(p)
    # Test-only provenance: fixture checkpoint must never ship as a trained surgical model.
    save_checkpoint(p,m,cfg,{"trained":True,"run_id":"synthetic-engineering-test"});loaded,ck=load_checkpoint(p)
    with torch.inference_mode():torch.testing.assert_close(m(torch.zeros(1,2,512))[0],loaded(torch.zeros(1,2,512))[0])
    predictor=Predictor(p);rgb=np.zeros((32,32,3),dtype=np.uint8);result=predictor.predict(rgb,0)
    assert abs(sum(result["probabilities"])-1)<1e-5
    with pytest.raises(ValueError):predictor.predict(rgb,0)
    predictor.reset();assert predictor.state is None;predictor.predict(rgb,0)
    ck["labels"]=[];torch.save(ck,p)
    with pytest.raises(ValueError):load_checkpoint(p)

def test_segmentation_and_metrics():
    ps=[{"video_time_seconds":i,"phase_id":c,"predicted_phase":str(c)} for i,c in enumerate([0,0,1,1,0])]
    seg=segments(ps,5);assert len(seg)==3;assert seg[-1]["end_seconds"]==5
    result=evaluate([0,0,1,1,0],[0,0,1,1,0]);assert result["accuracy"]==1;assert result["edit_score"]==100;assert result["segment_f1"]["0.5"]==1;assert result["boundary_mae_seconds"]==0
    bad=evaluate([0,0,1,1,0],[0,1,0,1,0]);assert bad["segment_count_error"]==2

def test_gradcam():
    from ml.inference.gradcam import gradcam
    cam=gradcam(small_model(),np.zeros((64,64,3),dtype=np.uint8),0)
    assert cam.shape==(64,64);assert np.isfinite(cam).all()

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cuda():
    m=small_model().cuda()
    with torch.inference_mode():assert m(torch.zeros(1,2,512,device="cuda"))[0].is_cuda

@pytest.mark.skipif(not hasattr(torch.backends,"mps") or not torch.backends.mps.is_available(),reason="MPS unavailable")
def test_mps():
    m=small_model().to("mps")
    with torch.inference_mode():assert m(torch.zeros(1,2,512,device="mps"))[0].device.type=="mps"
