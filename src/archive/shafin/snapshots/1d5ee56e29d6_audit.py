import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis_tools import final_synthesis as s
s.verify_inputs();roster=s.read('phase21_outputs/historical_test_overlap_roster.csv');expected=s.read('final_evidence/benchmark_audit.csv').set_index('stage');rc=lambda x:x.translate(str.maketrans('ACGT','TGCA'))[::-1];rows=[]
for stage,m in zip([1,2],s.manifests()):
    test=roster[roster.stage.eq(stage)];train=pd.concat([m[~m.role.eq('legacy_test')],m[m.record_id.isin(test[test.exact_training_match].record_id)]])
    exact=test.sequence.isin(set(train.sequence));reverse=test.sequence.map(rc).isin(set(train.sequence));candidate=m[m.role.eq('train_candidate')]
    for seq,digest in zip(m.sequence,m.sequence_sha256):
        import hashlib
        assert hashlib.sha256(seq.encode()).hexdigest()==digest
    counts=dict(stage=stage,train=len(train),test=len(test),exact=int(exact.sum()),reverse_complement=int(reverse.sum()),candidates=len(candidate),groups=candidate.similarity_group_id.nunique())
    for a,b in [('train','supplied_train_n'),('test','supplied_test_n'),('exact','exact_train_test_overlap'),('reverse_complement','reverse_complement_train_test_overlap'),('candidates','candidate_n'),('groups','candidate_groups')]:assert counts[a]==expected.loc[stage,b]
    rows.append(counts)
s.dump('audit_validation.json',dict(passed=True,counts=rows,full_sequence_hashes_checked=True))
print('Benchmark counts and DNA hashes reproduced.')
