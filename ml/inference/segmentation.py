def segments(predictions, end_seconds=None):
    result=[]
    for p in predictions:
        if not result or result[-1]["phase_id"]!=p["phase_id"]:
            if result:result[-1]["end_seconds"]=p["video_time_seconds"]
            result.append({"phase_id":p["phase_id"],"phase":p["predicted_phase"],"start_seconds":p["video_time_seconds"],"end_seconds":p["video_time_seconds"]})
    if result and end_seconds is not None:result[-1]["end_seconds"]=end_seconds
    return result
