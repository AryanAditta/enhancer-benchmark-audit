// Exhaustive, orientation-aware longest-common-subsequence audit for the
// 200 bp Stage 1 manifest. LCS is a necessary upper bound on the number of
// identical bases in any gapped pairwise alignment. This does not infer
// genomic coordinate identity or validate a local-identity threshold.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Record {
    std::string id, split, fold5, fold10, subtype, seq;
    std::array<std::array<uint64_t,4>,4> mask{};
};

int base(char c) {
    switch(c) { case 'A': return 0; case 'C': return 1;
        case 'G': return 2; case 'T': return 3; default: throw std::runtime_error("invalid base"); }
}

std::string revcomp(const std::string& s) {
    std::string out(s.size(), 'A');
    const char comp[] = {'T','G','C','A'};
    for(size_t i=0;i<s.size();++i) out[i]=comp[base(s[s.size()-1-i])];
    return out;
}

int lcs(const Record& a, const std::string& b) {
    uint64_t s[4] = {};
    for(char c:b) {
        uint64_t carry=1, borrow=0;
        for(int w=0;w<4;++w) {
            const uint64_t x=a.mask[base(c)][w]|s[w];
            const uint64_t shifted=(s[w]<<1)|carry;
            carry=s[w]>>63;
            const uint64_t diff=x-shifted-borrow;
            borrow=(x<shifted || (borrow && x==shifted));
            s[w]=x & ~diff;
        }
    }
    return __builtin_popcountll(s[0])+__builtin_popcountll(s[1])+
           __builtin_popcountll(s[2])+__builtin_popcountll(s[3]);
}

int main(int argc,char** argv) {
    if(argc!=3) {std::cerr<<"usage: audit records.tsv pair_results.tsv\n";return 2;}
    std::ifstream in(argv[1]);
    std::ofstream out(argv[2]);
    if(!in || !out) return 2;
    std::vector<Record> rows;
    while(true) {
        Record r;
        if(!(in>>r.id>>r.split>>r.fold5>>r.fold10>>r.subtype>>r.seq)) break;
        if(r.seq.size()!=200) throw std::runtime_error("not 200 bp");
        for(int i=0;i<200;++i) r.mask[base(r.seq[i])][i/64]|=uint64_t(1)<<(i%64);
        rows.push_back(std::move(r));
    }
    out<<"id_a\tid_b\tsplit_a\tsplit_b\tfold5_a\tfold5_b\tfold10_a\tfold10_b\tsubtype_a\tsubtype_b\tlcs_forward\tlcs_rc\n";
    uint64_t total=0,ge128=0,ge140=0,ge150=0,ge160=0;
    int overall=0,cross_test=0,cross_fold5=0,cross_fold10=0;
    for(size_t i=0;i<rows.size();++i) {
        for(size_t j=i+1;j<rows.size();++j) {
            const int fw=lcs(rows[i],rows[j].seq);
            const int rc=lcs(rows[i],revcomp(rows[j].seq));
            const int m=std::max(fw,rc);
            ++total;
            ge128+=(m>=128);ge140+=(m>=140);ge150+=(m>=150);ge160+=(m>=160);
            overall=std::max(overall,m);
            const bool crossing=rows[i].split!=rows[j].split;
            if(crossing) cross_test=std::max(cross_test,m);
            if(rows[i].split=="train" && rows[j].split=="train") {
                if(rows[i].fold5!=rows[j].fold5) cross_fold5=std::max(cross_fold5,m);
                if(rows[i].fold10!=rows[j].fold10) cross_fold10=std::max(cross_fold10,m);
            }
            if(m>=150) {
                out<<rows[i].id<<'\t'<<rows[j].id<<'\t'<<rows[i].split<<'\t'
                   <<rows[j].split<<'\t'<<rows[i].fold5<<'\t'<<rows[j].fold5
                   <<'\t'<<rows[i].fold10<<'\t'<<rows[j].fold10<<'\t'
                   <<rows[i].subtype<<'\t'<<rows[j].subtype<<'\t'
                   <<fw<<'\t'<<rc<<'\n';
            }
        }
        if(i%500==0) std::cerr<<"processed "<<i<<" / "<<rows.size()<<"\n";
    }
    std::cerr<<"total="<<total<<" max="<<overall<<" >=128="<<ge128
             <<" >=140="<<ge140<<" >=150="<<ge150<<" >=160="<<ge160
             <<" cross_test_max="<<cross_test<<" cross_fold5_max="<<cross_fold5
             <<" cross_fold10_max="<<cross_fold10<<"\n";
}
