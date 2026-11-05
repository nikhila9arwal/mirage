"""Known timings test the metric definitions without requiring a GPU."""
from pathlib import Path
import tempfile
import numpy as np
from analyze import analyze_lifecycle, concurrency, save

def main():
    with tempfile.TemporaryDirectory() as temporary:
        p=Path(temporary)
        passes=np.zeros((64,64,16),dtype=np.int64)
        tokens=np.zeros((64,64,4),dtype=np.int64)
        admissions=np.zeros((64,4),dtype=np.int64)
        history=np.full((64,64),-1,dtype=np.int64)
        history[:,:3]=[10,11,99]
        prompts=[]
        for request in range(64):
            slot=request%16
            admitted=((request//16)*1000+slot+1)*1_000_000_000
            admissions[request]=[admitted,slot,1,0]
            prompts.append(dict(request=request,length=4,token_ids=[1,2,3,4]))
            for j,position in enumerate((0,4,5)):
                r=passes[request,position]
                r[0]=admitted+(1_000_000_000,2_200_000_000,4_200_000_000)[j]
                r[1]=admitted+(2_000_000_000,4_000_000_000,7_000_000_000)[j]
                r[2:6]=[slot,j+1,4 if j==0 else 1,(10,11,99)[j]]
                r[6:10]=[1,2,3,4] if j==0 else [(10,11)[j-1],-1,-1,-1]
                r[10:12]=1
                tokens[request,j]=[admitted+(2_100_000_000,4_100_000_000,7_100_000_000)[j],
                                   (10,11,99)[j],slot,j+1]
        header=np.array([1_000_000_000,3_100_000_000_000,0,0,0,0,0,0],dtype=np.int64)
        np.savez(p/'measurements.npz',header=header,admissions=admissions,passes=passes,tokens=tokens)
        result=analyze_lifecycle(p,dict(prompts=prompts,eos_token_id=99),history,np.full(64,3))
        assert result['passed'],result['requests'][0]
        stats=result['statistics']
        assert abs(stats['ttft_from_admission_s']['mean']-2.1)<1e-10
        assert stats['all_intertoken_intervals_s']['mean']==2.5
        assert stats['per_request_mean_tpot_s']['mean']==2.5
        assert stats['intertoken_intervals_excluding_eos_s']['mean']==2.0
        busy=concurrency(np.array([[10_000_000_000,12_000_000_000],
                                   [11_000_000_000,13_000_000_000]],dtype=np.int64),0,20_000_000_000)
        assert busy['at_least_one_task_s']==3 and busy['mean_busy_workers']==.2
        save(Path(__file__).resolve().parents[1]/'review/analysis_tests.json',dict(passed=True,
            admission_ttft_s=2.1,tpot_s=2.5,excluding_eos_interval_s=2.,union_compute_s=3.,mean_workers=.2))
        print('PASS: TTFT, queue exclusion, TPOT, EOS, feedback, and worker interval union')

if __name__=='__main__':main()
