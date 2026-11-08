import asyncio,json,os,secrets,time,uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI,UploadFile,File,HTTPException,WebSocket,WebSocketDisconnect,Request,Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ml.datasets.preprocessing import OpenCVDecoder
from ml.datasets.cholec80 import PHASE_NAMES
from ml.inference.predictor import Predictor
from ml.inference.segmentation import segments
STORE=Path(os.getenv("SURGIVISION_STORAGE",".data"));STORE.mkdir(mode=0o700,parents=True,exist_ok=True)
CHECKPOINT=Path(os.getenv("SURGIVISION_CHECKPOINT","models/active.pt"));METRICS=Path(os.getenv("SURGIVISION_METRICS","models/metrics.json"))
LIMIT=int(os.getenv("SURGIVISION_UPLOAD_MB","1024"))*1024*1024;RETENTION=int(os.getenv("SURGIVISION_RETENTION_SECONDS","86400"));TOKEN=os.getenv("SURGIVISION_API_TOKEN","")
records={};pool=ThreadPoolExecutor(max_workers=1);active=set();tasks=set();inference_lock=asyncio.Lock()
def cleanup():
    for path in STORE.iterdir():
        if path.is_file() and path.stem not in active and time.time()-path.stat().st_mtime>RETENTION:
            path.unlink(missing_ok=True);records.pop(path.stem,None)
async def janitor():
    while True:cleanup();await asyncio.sleep(60)
@asynccontextmanager
async def lifespan(app):
    global pool,inference_lock
    pool=ThreadPoolExecutor(max_workers=1);inference_lock=asyncio.Lock();tasks.clear();active.clear()
    task=asyncio.create_task(janitor())
    yield
    task.cancel()
    for t in tasks:t.cancel()
    pool.shutdown(wait=False,cancel_futures=True)
app=FastAPI(title="SurgiVision AI Research API",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=os.getenv("SURGIVISION_ORIGINS","http://localhost:3000,http://127.0.0.1:3000").split(","),allow_methods=["GET","POST","DELETE"],allow_headers=["Authorization","Content-Type"])
def authorize(request:Request):
    if TOKEN and not secrets.compare_digest(request.headers.get("authorization",""),"Bearer "+TOKEN):raise HTTPException(401,"Authentication required")
def get_record(vid):
    cleanup()
    if vid not in records:raise HTTPException(404,"Video unavailable or expired")
    return records[vid]
def require_model():
    if not CHECKPOINT.is_file():raise HTTPException(503,"Model unavailable. Install a trained, provenance-checked checkpoint.")
@app.get("/api/health")
def health():return {"status":"ok","model_available":CHECKPOINT.is_file(),"research_only":True,"labels":PHASE_NAMES,"retention_seconds":RETENTION}
@app.get("/api/models",dependencies=[Depends(authorize)])
def models():return [{"id":"active","status":"checkpoint_present","metrics_available":METRICS.is_file()}] if CHECKPOINT.is_file() else []
@app.get("/api/models/{model_id}/metrics",dependencies=[Depends(authorize)])
def metrics(model_id:str):
    if model_id!="active" or not METRICS.is_file():raise HTTPException(404,"No measured evaluation report installed")
    return json.loads(METRICS.read_text())
@app.post("/api/videos/upload",dependencies=[Depends(authorize)])
async def upload(file:UploadFile=File(...),authorized:bool=False):
    if not authorized:raise HTTPException(422,"Confirm authorized, de-identified research footage")
    cleanup();suffix=Path(file.filename or "").suffix.lower()
    if suffix not in (".mp4",".avi",".mov",".mkv"):raise HTTPException(415,"Unsupported video format")
    vid=uuid.uuid4().hex;path=STORE/(vid+suffix);size=0
    try:
        with path.open("xb") as f:
            os.chmod(path,0o600)
            while True:
                chunk=await file.read(1024*1024)
                if not chunk:break
                size+=len(chunk)
                if size>LIMIT:raise HTTPException(413,"Upload exceeds configured limit")
                f.write(chunk)
        decoder=await asyncio.to_thread(OpenCVDecoder,path);duration=decoder.duration;fps=decoder.fps;decoder.close()
    except HTTPException:path.unlink(missing_ok=True);raise
    except Exception:path.unlink(missing_ok=True);raise HTTPException(422,"Video cannot be decoded")
    finally:await file.close()
    records[vid]={"id":vid,"path":str(path),"name":Path(file.filename).name,"duration_seconds":duration,"source_fps":fps,"size_bytes":size,"state":"uploaded","predictions":[],"segments":[]}
    return public(records[vid])
def public(record):return {k:v for k,v in record.items() if k!="path"}
def analyze_sync(vid):
    r=records[vid]
    try:
        predictor=Predictor(CHECKPOINT,streaming=False);pred=[];started=time.perf_counter()
        for _,t,rgb in OpenCVDecoder(r["path"],predictor.sample_fps):
            pred.append(predictor.predict(rgb,t));r["processed_seconds"]=t
        r.update(state="completed",predictions=pred,segments=segments(pred,r["duration_seconds"]),processing_seconds=time.perf_counter()-started)
    except Exception as e:r.update(state="failed",error=str(e))
    finally:active.discard(vid)
async def run_analysis(vid):
    async with inference_lock:await asyncio.get_running_loop().run_in_executor(pool,analyze_sync,vid)
@app.post("/api/videos/{vid}/analyze",dependencies=[Depends(authorize)],status_code=202)
async def analyze(vid:str):
    r=get_record(vid);require_model()
    if r["state"] in ("queued","processing"):raise HTTPException(409,"Analysis already active")
    if len(tasks)>=4:raise HTTPException(429,"Analysis queue full")
    r["state"]="queued";active.add(vid)
    async def job():
        try:
            async with inference_lock:
                r["state"]="processing";await asyncio.get_running_loop().run_in_executor(pool,analyze_sync,vid)
        except Exception as e:r.update(state="failed",error=str(e))
        finally:active.discard(vid)
    task=asyncio.create_task(job());tasks.add(task);task.add_done_callback(tasks.discard)
    return {"state":"queued","id":vid}
@app.get("/api/videos/{vid}/results",dependencies=[Depends(authorize)])
def results(vid:str):return public(get_record(vid))
@app.get("/api/videos/{vid}/timeline",dependencies=[Depends(authorize)])
def timeline(vid:str):return {"segments":get_record(vid)["segments"]}
@app.get("/api/videos/{vid}/content",dependencies=[Depends(authorize)])
def content(vid:str):return FileResponse(get_record(vid)["path"])
@app.delete("/api/videos/{vid}",dependencies=[Depends(authorize)])
def delete(vid:str):
    r=get_record(vid)
    if vid in active:raise HTTPException(409,"Stop processing or streaming before deletion")
    Path(r["path"]).unlink(missing_ok=True);records.pop(vid);return {"deleted":True}
def next_sample(decoder):
    try:return next(decoder)
    except StopIteration:return None
@app.websocket("/api/stream/{vid}")
async def stream(ws:WebSocket,vid:str):
    await ws.accept()
    # Authenticate in first message, never put tokens into URLs/logs.
    try:
        hello=await asyncio.wait_for(ws.receive_json(),15)
        if TOKEN and not secrets.compare_digest(str(hello.get("token","")),TOKEN):await ws.close(code=1008);return
        r=get_record(vid);require_model()
        if vid in active or inference_lock.locked():await ws.send_json({"error":"Inference service busy"});await ws.close();return
        async with inference_lock:
            active.add(vid);predictor=await asyncio.to_thread(Predictor,CHECKPOINT);decoder=iter(OpenCVDecoder(r["path"],predictor.sample_fps));paused=True;target=0.;history=[]
            try:
                await ws.send_json({"state":"ready"})
                while True:
                    if paused:cmd=await ws.receive_json()
                    else:
                        try:cmd=await asyncio.wait_for(ws.receive_json(),.001)
                        except asyncio.TimeoutError:cmd=None
                    if cmd:
                        action=cmd.get("action")
                        if action=="stop":break
                        if action=="pause":paused=True;continue
                        if action in ("seek","reset"):
                            target=max(0.,min(float(cmd.get("time",0)),r["duration_seconds"]));decoder.close();decoder=iter(OpenCVDecoder(r["path"],predictor.sample_fps));predictor.reset();history=[]
                            # Replay the prefix to rebuild causal hidden state. No future frames.
                            await ws.send_json({"state":"warming","target":target})
                        if action in ("resume","seek","reset"):paused=False
                    if paused:continue
                    sample=await asyncio.to_thread(next_sample,decoder)
                    if sample is None:await ws.send_json({"state":"completed"});break
                    _,t,rgb=sample;started=time.perf_counter();prediction=await asyncio.to_thread(predictor.predict,rgb,t);history.append(prediction)
                    if t<target:continue
                    await ws.send_json({"prediction":prediction,"segments":segments(history,t),"lag_seconds":max(0,time.perf_counter()-started-1./predictor.sample_fps)})
                    await asyncio.sleep(max(0.,1./predictor.sample_fps-(time.perf_counter()-started)))
            finally:decoder.close();active.discard(vid)
    except (WebSocketDisconnect,asyncio.TimeoutError):pass
    except HTTPException as e:await ws.send_json({"error":e.detail});await ws.close()
    except Exception:await ws.send_json({"error":"Streaming failed; verify checkpoint and video"});await ws.close()

@app.post("/api/videos/{vid}/gradcam",dependencies=[Depends(authorize)])
async def attribution(vid:str,seconds:float=0.):
    r=get_record(vid);require_model()
    if not 0<=seconds<r["duration_seconds"]:raise HTTPException(422,"Timestamp outside video")
    if vid in active or inference_lock.locked():raise HTTPException(409,"Inference service busy")
    def compute():
        from ml.inference.gradcam import gradcam
        predictor=Predictor(CHECKPOINT);decoder=iter(OpenCVDecoder(r["path"],predictor.sample_fps));selected=None;prior=None
        try:
            for _,t,rgb in decoder:
                if t>seconds:break
                prior=predictor.state
                result=predictor.predict(rgb,t);selected=(rgb,result,prior)
            if selected is None:raise ValueError("No frame at timestamp")
            rgb,result,prior=selected
            heat=gradcam(predictor.model,rgb,result["phase_id"],prior)
            import cv2,base64
            # RGB -> BGR for OpenCV PNG encoding; display is an exploratory overlay.
            color=cv2.applyColorMap((heat*255).astype("uint8"),cv2.COLORMAP_JET)
            overlay=cv2.addWeighted(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),.6,color,.4,0)
            ok,encoded=cv2.imencode(".png",overlay)
            if not ok:raise ValueError("Overlay encoding failed")
            return {"video_time_seconds":result["video_time_seconds"],"phase_id":result["phase_id"],"overlay":"data:image/png;base64,"+base64.b64encode(encoded).decode(),"caveat":"Exploratory current-frame attribution conditioned on past state; not a causal explanation or safety assessment."}
        finally:decoder.close()
    async with inference_lock:
        active.add(vid)
        try:return await asyncio.to_thread(compute)
        except Exception:raise HTTPException(422,"Attribution failed; verify compatible checkpoint and video")
        finally:active.discard(vid)
