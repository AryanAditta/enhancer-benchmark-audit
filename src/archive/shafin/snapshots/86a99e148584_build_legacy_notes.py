from pathlib import Path
import json, html, re, hashlib, collections

ROOT = Path.cwd() / 'research_notes/Enhancer manuscript citation corrections'
SOURCES = ROOT / 'legacy_sources'
STANDARD = {'train_enhancer': 1484, 'train_non_enhancer': 1484,
            'train_strong': 742, 'train_weak': 742,
            'test_enhancer': 200, 'test_non_enhancer': 200,
            'test_strong': 100, 'test_weak': 100, 'sequence_length_bp': 200}
GLOBAL_LIMIT = ('Published benchmark lineage is supported; byte identity with our supplied local files has not been established. '
                'No source in this review independently establishes that this paper trained on our exact duplicated records.')
entries = []

def add(id, model, doi, year, url, evidence, reading_scope, role='published_fixed_test_lineage',
        limits='', counts=None, source_files=(), location='', include=True, extra_urls=()):
    entries.append(dict(id=id, model=model, doi=doi, year=year, primary_url=url,
                        supporting_urls=[url, *extra_urls], evidence_paraphrase=evidence,
                        reading_scope=reading_scope, benchmark_role=role,
                        limitations=' '.join([GLOBAL_LIMIT, limits]).strip(),
                        counts=dict(STANDARD) if counts is None else counts,
                        local_source_files=list(source_files), source_location=location,
                        core_twenty_candidate=include,
                        exact_local_file_byte_identity_verified=False))

add('ienhancer_2l_2016', 'iEnhancer-2L', '10.1093/bioinformatics/btv604', 2016,
    'https://academic.oup.com/bioinformatics/article/32/3/362/1744331?searchresult=1',
    'The original two-layer predictor defines the 200 bp enhancer/strength benchmark. Its evaluation section emphasizes jackknife testing, where one benchmark sample is held out in turn. This is the origin citation, not direct evidence that the original article reported a separate 400-record independent test.',
    'Primary publisher indexed evaluation section; publisher metadata. Later benchmark-origin descriptions checked against primary full-text descendants.',
    role='benchmark_origin_not_fixed_test_verified',
    limits='Do not write that all later fixed-test protocols were already implemented in this original article. Publication year is 2016, although first online in 2015.',
    counts={**STANDARD, 'test_enhancer': None, 'test_non_enhancer': None, 'test_strong': None, 'test_weak': None},
    location='Evaluation/jackknife section')

add('ienhancer_el_2018', 'iEnhancer-EL', '10.1093/bioinformatics/bty458', 2018,
    'https://academic.oup.com/bioinformatics/article/34/22/3835/5034432',
    'Section 3.2 explicitly evaluates an independent dataset constructed with the benchmark protocol: 100 strong enhancers, 100 weak enhancers and 200 non-enhancers. The authors state that independent samples do not occur in training. Later primary papers attribute the commonly reused independent set to this article.',
    'Primary publisher full-text section 3.2 exposed by indexing and publisher PDF excerpt; full local HTML request was access-blocked.',
    limits='The paper asserts independence; our local-copy audit challenges independence of supplied copies only and does not authenticate the files used by these authors.',
    location='Section 3.2; independent dataset in Supplementary Information S4')

add('ienhancer_5step_2019', 'iEnhancer-5Step', '10.1016/j.ab.2019.02.017', 2019,
    'https://doi.org/10.1016/j.ab.2019.02.017',
    'The Methods reuse the Liu benchmark: 1,484 enhancers and 1,484 non-enhancers, with 742 strong and 742 weak enhancers. An independent dataset contains 200 enhancers (100 per strength class) and 200 non-enhancers. Five-fold training-set evaluation and independent evaluation are described separately.',
    'Full primary research paper from a coauthor-hosted publisher PDF; Methods 2.1, 2.6 and independent-test results read.',
    source_files=['ienhancer_5step.pdf'], location='Sections 2.1, 2.6 and 3.3',
    extra_urls=['https://ivyhuiyuanyeh.com/wp-content/uploads/2019/10/le-2019-ienhancer-step-identifying-enhancer.pdf'])

add('ienhancer_rf_2021', 'iEnhancer-RF', '10.1016/j.chemolab.2021.104284', 2021,
    'https://www.sciencedirect.com/science/article/pii/S0169743921000526',
    'The primary publisher text describes a random-forest enhancer/strength predictor, chromatin-state data from nine cell lines, and both cross-validation and independent tests. The abstract reports improvements in independent-test accuracy. The accessible Methods excerpt does not expose the complete sample counts.',
    'Primary abstract plus partial publisher Introduction/Materials/results excerpt; full Methods and count table unavailable.',
    role='independent_test_reported_counts_not_primary_verified',
    limits='Counts are not inferred from secondary comparison tables. Cite for the independent-test literature context, not as independently verified exact-count/file provenance.',
    counts={k: None for k in STANDARD}, location='Publisher abstract, Materials and results introduction')

add('ienhancer_deep_2022', 'iEnhancer-Deep', '10.3390/app12042120', 2022,
    'https://www.mdpi.com/2076-3417/12/4/2120',
    'The one-hot/CNN study uses the established two-stage benchmark and explicitly tests on an independent dataset of 100 strong enhancers, 100 weak enhancers and 200 non-enhancers. Its independent-test section compares both enhancer identification and strength prediction.',
    'Full primary publisher PDF; dataset and independent-test sections read.',
    source_files=['ienhancer_deep.pdf'], location='Section 3.4; independent comparisons Tables 7-8',
    extra_urls=['https://mdpi-res.com/d_attachment/applsci/applsci-12-02120/article_deploy/applsci-12-02120.pdf'])

add('ienhancer_ecnn_2019', 'iEnhancer-ECNN', '10.1186/s12864-019-6336-3', 2019,
    'https://link.springer.com/article/10.1186/s12864-019-6336-3',
    'The Methods attribute the data to Liu 2016/2018 and list the standard development and independent counts. Five CNNs fitted on stratified folds are combined into an ensemble, then tested on the independent set; the entire partition/train/test procedure is repeated ten times.',
    'Full primary JATS article downloaded from Europe PMC; dataset, ensemble and trial descriptions read.',
    limits='This prior method itself ensembles several CNNs; do not imply that all historical comparators were single models.',
    source_files=['PMC6929481.xml'], location='Methods: dataset and ensemble procedure')

add('ienhancer_eblstm_2021', 'iEnhancer-EBLSTM', '10.3389/fgene.2021.665498', 2021,
    'https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2021.665498/full',
    'The dataset is attributed to Liu 2016/2018 and Nguyen 2019, with 1,484 enhancer and 1,484 negative development samples. Enhancers divide equally into 742 strong and 742 weak. The independent set has 200 enhancers (100 strong, 100 weak) and 200 negatives; the work evaluates an ensemble of bidirectional LSTMs.',
    'Full primary JATS article; dataset and independent evaluation descriptions read.',
    limits='The authors use an ensemble, so its computation/aggregation differs from single-model baselines.',
    source_files=['PMC8021722.xml'], location='Dataset; independent evaluation')

add('ienhancer_gan_2021', 'iEnhancer-GAN', '10.3390/ijms22073589', 2021,
    'https://www.mdpi.com/1422-0067/22/7/3589',
    'Methods 3.1 use the Liu training benchmark and attribute the 400-sequence independent set to iEnhancer-EL. The paper specifies 1,484 enhancer plus 1,484 non-enhancer training sequences, and 100 strong plus 100 weak plus 200 non-enhancer independent sequences. GAN augmentation is applied to training.',
    'Full primary JATS article; Methods 3.1 and independent evaluation read.',
    limits='Training augmentation changes the recipe; common source/counts do not imply identical training budgets.',
    source_files=['PMC8036415.xml'], location='Section 3.1 and independent-test results')

add('ienhancer_dcla_2022', 'iEnhancer-DCLA', '10.1186/s12859-022-05033-x', 2022,
    'https://link.springer.com/article/10.1186/s12859-022-05033-x',
    'Methods explicitly reuse the Liu benchmark, listing 1,484 enhancers and 1,484 non-enhancers in training, and 200 enhancers (100 strong, 100 weak) plus 200 non-enhancers in the independent test. The model combines dna2vec, convolution, bidirectional LSTM and attention.',
    'Full primary JATS article; benchmark and architecture sections read.',
    source_files=['PMC9664816.xml'], location='Methods: benchmark dataset and model architecture')

add('ienhancer_dcsv_2023', 'iEnhancer-DCSV', '10.3389/fgene.2023.1132018', 2023,
    'https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2023.1132018/full',
    'Methods 2.1 describe the Liu benchmark and give the standard 2,968 training records and 400 independent records, including balanced strong/weak subsets. The DenseNet/attention model is compared to published methods on this independent set.',
    'Full primary JATS article; dataset and independent-comparison sections read.',
    limits='The prose has inconsistent wording about redundancy removal; it is not evidence that our own LCS rule equals the published CD-HIT screen.',
    source_files=['PMC10014624.xml'], location='Methods 2.1; Table 1; independent-comparison Table 6')

add('ienhancer_dcsa_2023', 'iEnhancer-DCSA', '10.1186/s12864-023-09468-1', 2023,
    'https://link.springer.com/article/10.1186/s12864-023-09468-1',
    'The primary Methods list 1,484 enhancers and 1,484 non-enhancers for development, with 742 examples in each strength class. The independent dataset consists of 100 strong, 100 weak and 200 non-enhancers. The model is trained with the development set and evaluated independently; additional random-seed experiments are described.',
    'Full primary JATS article; dataset and independent-test sections read.',
    source_files=['PMC10339552.xml'], location='Dataset; independent-test results and Table 4')

add('ienhancer_elm_2023', 'iEnhancer-ELM', '10.1093/bioadv/vbad043', 2023,
    'https://doi.org/10.1093/bioadv/vbad043',
    'This language-model study includes Liu and Basith datasets. For Liu it states 1,484 enhancers plus 1,484 non-enhancers in training and 200 plus 200 in testing. It explicitly evaluates enhancer identification only, not strong-versus-weak classification.',
    'Full primary JATS article; dataset and study-scope sections read.',
    limits='Use as Stage 1 lineage evidence only; do not claim the paper evaluated the legacy Stage 2 independent set.',
    counts={**STANDARD, 'train_strong': None, 'train_weak': None, 'test_strong': None, 'test_weak': None},
    source_files=['PMC10125906.xml'], location='Dataset section',
    extra_urls=['https://pmc.ncbi.nlm.nih.gov/articles/PMC10125906/'])

add('enhancer_lstmatt_2022', 'Enhancer-LSTMAtt', '10.3390/biom12070995', 2022,
    'https://www.mdpi.com/2218-273X/12/7/995',
    'The primary article reuses the Liu benchmark and attributes its independent set to iEnhancer-EL. It lists 742 strong, 742 weak and 1,484 non-enhancer development sequences. Its prose lists 100 strong, 100 weak and only 100 non-enhancer independent samples, while describing comparison with the shared published independent-test literature.',
    'Full primary JATS article; dataset, evaluation and comparison caveats read.',
    role='published_lineage_with_count_ambiguity',
    limits='The 100 non-enhancer statement is inconsistent with the standard 200-negative protocol. Do not silently correct it or claim exact matching independent counts. Its paper also explicitly says most comparison-method scores were taken from published papers rather than rerun.',
    counts={**STANDARD, 'test_non_enhancer': 100, 'test_count_warning': 'Primary prose lists 100 non-enhancers, not 200; unresolved in this review.'},
    source_files=['PMC9313278.xml'], location='Dataset section and independent comparison/Table 6')

add('capsenhancer_2024', 'CapsEnhancer', '10.1021/acs.jcim.4c00546', 2024,
    'https://pubs.acs.org/doi/10.1021/acs.jcim.4c00546',
    'The primary article lists 1,484 enhancer and 1,484 non-enhancer development records, splitting enhancers into 742 strong and 742 weak. It attributes the independent set to Liu and specifies 100 strong, 100 weak and 200 non-enhancer sequences, then reports both independent-test tasks.',
    'Full primary JATS article; dataset and independent comparison tables read.',
    source_files=['PMC11267569.xml'], location='Dataset section and independent Tables 3-4')

add('ienhancer_sknn_2023', 'iEnhancer-SKNN', '10.1093/bfgp/elac057', 2023,
    'https://academic.oup.com/bfg/article/22/3/302/7008796',
    'The publisher Methods describe the nine-cell Liu data and list the standard counts: 2,968 training and 400 independent samples for enhancer identification, and 1,484 training and 200 independent samples for strength. Stacking is trained using development predictions and then produces independent predictions.',
    'Primary publisher indexed Methods and algorithm steps; local publisher download was access-blocked.',
    location='Materials and Methods: Dataset; stacking procedure')

add('bert_enhancer_2021', 'BERT-Enhancer / BERT-2D CNN', '10.1093/bib/bbab005', 2021,
    'https://academic.oup.com/bib/article/22/5/bbab005/6128847',
    'The primary publisher Methods retrieve the iEnhancer-2L lineage benchmark, with 1,484 enhancer plus 1,484 non-enhancer development records and 200 plus 200 independent records. A BERT-based representation is combined with a 2D convolutional classifier for enhancer identification.',
    'Primary publisher indexed benchmark Methods and abstract; local publisher download access-blocked.',
    limits='The DOI is bbab005. Some later reference lists print bbaa398 for the same title; use verified publisher/Crossref metadata. The reading supports Stage 1, not a verified strength-task evaluation.',
    counts={**STANDARD, 'train_strong': None, 'train_weak': None, 'test_strong': None, 'test_weak': None},
    location='Materials and Methods: benchmark dataset')

add('ienhancer_dlra_2022', 'iEnhancer-DLRA', '10.1093/bfgp/elac023', 2022,
    'https://academic.oup.com/bfg/article/21/5/399/6658443',
    'Publisher Methods 2.1 explicitly list 1,484 enhancers and 1,484 non-enhancers in development, equal strength-class counts, and 100 strong plus 100 weak plus 200 non-enhancers in the independent set. They state that design/parameter optimization uses training data and the independent set evaluates comparisons.',
    'Primary publisher indexed Methods 2.1 and abstract; full local article unavailable.',
    location='Section 2.1 benchmark dataset')

add('ienhancer_cnn_2020', 'iEnhancer-CNN', '10.1109/ACCESS.2020.2982666', 2020,
    'https://doi.org/10.1109/ACCESS.2020.2982666',
    'The primary article Methods II.B lists 1,484 enhancers and 1,484 non-enhancers, with 742 strong and 742 weak enhancers. It reuses the Liu benchmark and independent data, specifying 100 strong, 100 weak and 200 non-enhancers. The independent-test results are reported in section III.C.',
    'Author-uploaded full primary paper on ResearchGate, including publisher DOI and text; Crossref metadata independently checked. Publisher endpoint itself unavailable.',
    limits='The author-uploaded article is the primary research content; do not present ResearchGate as an independent dataset audit.',
    location='Methods II.B; Results III.C',
    extra_urls=['https://www.researchgate.net/publication/340107905_Identifying_Enhancers_and_Their_Strength_by_the_Integration_of_Word_Embedding_and_Convolution_Neural_Network'])

add('adh_enhancer_2024', 'ADH-Enhancer', '10.1093/bib/bbae030', 2024,
    'https://doi.org/10.1093/bib/bbae030',
    'The primary article reuses the Liu benchmark for enhancer and strength prediction, describes a 2,968-record core dataset and independent testing. Figure 4 labels 1,484 enhancer plus 1,484 non-enhancer core records and 200 enhancer plus 200 non-enhancer independent records, with equal strong/weak subclasses.',
    'Full primary JATS article plus publisher Figure 4 description; dataset and evaluation sections read.',
    source_files=['PMC10885011.xml'], location='Benchmark datasets; Figure 4',
    extra_urls=['https://pmc.ncbi.nlm.nih.gov/articles/PMC10885011/'])

add('dnabert2_enhancer_2025', 'DNABERT2-Enhancer', '10.1371/journal.pone.0320085', 2025,
    'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0320085',
    'The study already fine-tunes DNABERT-2 for enhancer and strength classification. Its Liu benchmark has 1,484 enhancers (742 strong, 742 weak) plus 1,484 non-enhancers, and an independent set of 200 enhancers (100 per strength) plus 200 non-enhancers. Basith cell-specific datasets are assessed separately.',
    'Full primary publisher XML downloaded; dataset and model descriptions read.',
    limits='This is direct prior DNABERT-2 enhancer-classification work. Our local grouped/repeated comparisons and MPRA audit should not be presented as the first application of DNABERT-2 to enhancer identification.',
    source_files=['dnabert2_enhancer_2025.xml'], location='Benchmark dataset and proposed model')

# Four relevant caution records: useful literature context, not extra papers to pad the core set.
add('enhancerpred_2016', 'EnhancerPred', '10.1038/srep38741', 2016,
    'https://www.nature.com/articles/srep38741',
    'The primary Methods use the recently constructed Liu benchmark, containing 742 strong enhancers, 742 weak enhancers and 1,484 non-enhancers. This substantiates early reuse of the benchmark, but the Methods read here do not establish use of the later fixed 400-record independent set.',
    'Full primary JATS article; original dataset and evaluation descriptions read.',
    role='benchmark_lineage_not_fixed_test_verified',
    limits='Later papers reproduce independent-test scores under this method name; that is not proof that the original 2016 article itself evaluated the 2018 independent set.',
    counts={**STANDARD, 'test_enhancer': None, 'test_non_enhancer': None, 'test_strong': None, 'test_weak': None},
    source_files=['PMC5150536.xml'], location='Methods dataset', include=False)

add('ienhancer_xg_2021', 'iEnhancer-XG', '10.1093/bioinformatics/btaa914', 2021,
    'https://academic.oup.com/bioinformatics/article/37/8/1060/5942971?searchresult=1',
    'The publisher Methods describe the standard development and independent sequence counts, but explicitly state that training and independent data were merged for model building. The article also discusses 10-fold evaluation on independent data. It is part of the benchmark lineage with a materially different evaluation description.',
    'Primary publisher indexed Methods 2.1 and evaluation excerpts; original publisher page text inspected.',
    role='shared_lineage_different_evaluation_protocol',
    limits='Do not group this paper under a claim that every cited study used the original independent set as an untouched holdout. This review does not resolve how every published table was generated.',
    location='Methods 2.1; independent-dataset evaluation', include=False)

add('ienhancer_mfgbdt_2021', 'iEnhancer-MFGBDT', '10.3934/mbe.2021434', 2021,
    'https://www.aimspress.com/article/doi/10.3934/mbe.2021434',
    'The primary paper uses the 2,968-record Liu benchmark and separately describes 400 independent sequences (100 strong, 100 weak, 200 non-enhancers). Section 3.5 explicitly reports 10-fold cross-validation on that independent dataset, and Table 4 repeats this evaluation description.',
    'Full primary publisher PDF; Methods and section 3.5/Table 4 read.',
    role='shared_lineage_different_evaluation_protocol',
    limits='A common dataset name and count do not make this an identical held-out evaluation to studies that fit only on the development dataset.',
    source_files=['ienhancer_mfgbdt.pdf'], location='Section 3.5; Table 4', include=False,
    extra_urls=['https://www.aimspress.com/aimspress-data/mbe/2021/6/PDF/mbe-18-06-434.pdf'])

add('deepdualenhancer_2024', 'DeepDualEnhancer', '10.3390/ijms252111744', 2024,
    'https://www.mdpi.com/1422-0067/25/21/11744',
    'The primary dataset section names the Liu benchmark and lists 1,484 enhancer plus 1,484 non-enhancer training examples. It lists 100 strong, 100 weak and 100 non-enhancer independent examples and says the test set was also processed by CD-HIT.',
    'Full primary JATS article; dataset section read.',
    role='published_lineage_with_count_and_processing_ambiguity',
    limits='The 100-negative prose and stated additional test-set filtering prevent a simple claim of an unmodified standard 400-record split. Do not silently substitute the customary 200 negatives.',
    counts={**STANDARD, 'test_non_enhancer': 100, 'test_count_warning': 'Primary prose lists 100 non-enhancers; additional CD-HIT test processing stated.'},
    source_files=['PMC11546905.xml'], location='Dataset section; Tables 5-6', include=False)

def clean(value):
    return re.sub('<[^>]+>', '', html.unescape(value or '')).strip()

for e in entries:
    meta_path = SOURCES / (e['doi'].replace('/', '_') + '.crossref.json')
    m = json.loads(meta_path.read_text(encoding='utf8'))
    e.update(title=clean(m['title'][0]), journal=clean(m.get('container-title', [''])[0]),
             volume=m.get('volume', ''), issue=m.get('issue', ''),
             pages=m.get('page', ''), article_number=m.get('article-number', ''),
             authors=[{k: a.get(k, '') for k in ('family', 'given')} for a in m.get('author', [])],
             metadata_url='https://api.crossref.org/works/' + e['doi'],
             metadata_source_file=str(meta_path.relative_to(ROOT)).replace('\\', '/'),
             supported_claim=e['evidence_paraphrase'],
             confidence=('high for the stated published-method evidence; exact local byte identity unverified'
                         if 'ambiguity' not in e['benchmark_role'] and 'not_primary_verified' not in e['benchmark_role']
                         else 'limited; count/protocol uncertainty explicitly retained'))

payload = dict(topic='Legacy enhancer benchmark citation correction', checked_date='2026-10-15',
               scope='20 core candidate papers plus 4 protocol/origin warnings. No exact local file provenance is inferred from publication lineage.',
               recommended_wording=(
                   'The 200 bp classification benchmark introduced with iEnhancer-2L and the independent set described with iEnhancer-EL have been repeatedly reused in published enhancer predictors. '
                   'Examples span engineered features, CNN/RNN ensembles and language-model methods. '
                   'Our overlap finding concerns the supplied copies audited here; we did not authenticate the exact sequence files used in every cited publication.'),
               prohibited_wording=[
                   'All cited papers used our leaky files.',
                   'All historical scores are inflated by the same amount.',
                   'Every paper followed an identical independent-test protocol.',
                   'Our LCS >=160 connected-component screen reproduces published CD-HIT filtering.',
                   'The first DNABERT-2 enhancer classifier is introduced here.'],
               cautions=[
                   'Distinguish lineage from exact file identity, and file overlap from demonstrated causal inflation.',
                   'The original iEnhancer-2L emphasizes jackknife evaluation; the commonly reused independent set is explicitly described in iEnhancer-EL.',
                   'Some papers report 20% versus 80% similarity inconsistently. Preserve the cited authors’ description rather than treating those statements as a validated common filtering operation.',
                   'The bibliography should reflect scientific relevance, not an arbitrary reference-count target.'],
               role_counts=dict(collections.Counter(e['benchmark_role'] for e in entries)), entries=entries)
(ROOT / 'legacy_benchmark.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf8')

lines = ['# Published use of the legacy 200 bp enhancer benchmark', '',
         '## What can the manuscript safely say about the benchmark lineage?', '',
         '### Takeaway', payload['recommended_wording'], '',
         '### Cited Findings',
         '- The benchmark-origin article emphasizes jackknife evaluation; the later iEnhancer-EL explicitly describes a separate 100-strong/100-weak/200-negative independent set. '
         '— [iEnhancer-2L](https://doi.org/10.1093/bioinformatics/btv604); [iEnhancer-EL](https://doi.org/10.1093/bioinformatics/bty458)',
         '- Reuse spans ensemble CNNs, word embeddings, convolution/attention frameworks and DNABERT-2. '
         '— [iEnhancer-ECNN](https://doi.org/10.1186/s12864-019-6336-3); [iEnhancer-5Step](https://doi.org/10.1016/j.ab.2019.02.017); '
         '[iEnhancer-DLRA](https://doi.org/10.1093/bfgp/elac023); [DNABERT2-Enhancer](https://doi.org/10.1371/journal.pone.0320085)',
         '', '### Inferences',
         '- Repeated use makes auditing the supplied benchmark scientifically relevant. Publication names and matching sample counts establish a lineage, not byte identity. '
         'The exact local overlap finding remains an observation about this study’s authenticated files.',
         '', '### Gaps',
         '- No cited authors’ historical run files were byte-authenticated in this literature review; no claim of individual misconduct or common numerical score inflation is supported.',
         '', '## Which primary papers support the literature context?', '',
         '### Takeaway',
         'Twenty core candidate papers are listed below, with four additional records that explain origin/protocol ambiguities. '
         'Use the primary Methods citations for lineage claims and keep limitations attached. This is not a request to cite every record.', '',
         '### Cited Findings']
for i, e in enumerate(entries, 1):
    authors = '; '.join(a['family'] + ', ' + a['given'] for a in e['authors'])
    page = e['pages'] or e['article_number']
    tag = 'core candidate' if e['core_twenty_candidate'] else 'additional caution/context'
    lines += [f"\n#### {i}. {e['model']} ({e['year']}; {tag})",
              f"{authors}. {e['title']}. *{e['journal']}* {e['volume']}({e['issue']}):{page}. DOI: {e['doi']}. — [Primary source]({e['primary_url']})",
              f"- Evidence: {e['evidence_paraphrase']} — [Source]({e['supporting_urls'][-1]})",
              '- Counts as read: `' + json.dumps(e['counts'], ensure_ascii=False) + '`.',
              f"- Reading scope: {e['reading_scope']} Source location: {e['source_location']}.",
              f"- Limits: {e['limitations']}"]
lines += ['', '### Inferences',
          '- Use a compact citation cluster in the Introduction and a benchmark-lineage table in the supplement rather than an unreadable long list in every result.',
          '- Separate the benchmark-origin citation, standard independent-test users, Stage 1-only users and differently described evaluations.',
          '', '### Gaps',
          '- iEnhancer-RF: accessible primary source confirms independent testing but the full count table was unavailable, so sample counts are left null.',
          '- iEnhancer-EL, iEnhancer-SKNN, BERT-Enhancer and iEnhancer-DLRA: method passages were available in publisher indexing but local publisher HTML downloads were blocked; this is not described as a complete downloaded full-text read.',
          '', '## What cautions should govern the revised claim?', '',
          '### Takeaway',
          'The defensible conclusion is widespread reuse of a benchmark lineage plus overlap in our supplied copies. '
          'The literature does not establish that all published runs used the exact same contaminated bytes or would suffer the same metric change.', '',
          '### Cited Findings',
          '- iEnhancer-XG states that training and independent data were merged for model building. '
          '— [Primary Methods 2.1](https://academic.oup.com/bioinformatics/article/37/8/1060/5942971?searchresult=1)',
          '- iEnhancer-MFGBDT describes 10-fold cross-validation on the independent dataset. '
          '— [Primary section 3.5, Table 4](https://www.aimspress.com/aimspress-data/mbe/2021/6/PDF/mbe-18-06-434.pdf)',
          '- Enhancer-LSTMAtt attributes its independent data to iEnhancer-EL but lists only 100 non-enhancers in prose. '
          '— [Primary dataset section](https://pmc.ncbi.nlm.nih.gov/articles/PMC9313278/)',
          '- DeepDualEnhancer also lists only 100 non-enhancers in prose and describes CD-HIT processing of its independent set. '
          '— [Primary dataset section](https://pmc.ncbi.nlm.nih.gov/articles/PMC11546905/)',
          '- DNABERT2-Enhancer already applies DNABERT-2 to these classification tasks. '
          '— [Primary paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0320085)',
          '', '### Inferences',
          '- A transparent supplemental lineage table is more useful than describing every past method as an identical untouched-test comparison.',
          '- Historical single-model, multi-fold ensemble and multi-seed ensemble results should not be treated as equal-computation experiments.',
          '', '### Gaps',
          '- The exact fixed-test content used by each historical publication and the size of any resulting performance inflation remain unverified.',
          '- This citation correction does not rerun historical models or change locked experiments.', '']
(ROOT / 'legacy_benchmark.md').write_text('\n'.join(lines), encoding='utf8')
manifest = []
for path in sorted(SOURCES.iterdir()):
    if path.is_file():
        data = path.read_bytes()
        manifest.append(dict(path=str(path.relative_to(ROOT)).replace('\\','/'), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
(ROOT / 'legacy_source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
assert len([e for e in entries if e['core_twenty_candidate']]) == 20
assert len({e['doi'] for e in entries}) == len(entries)
assert all(e['authors'] and e['title'] and e['journal'] for e in entries)
assert all(not e['exact_local_file_byte_identity_verified'] for e in entries)
print(json.dumps({'entries': len(entries), 'core_candidates': 20, 'role_counts': payload['role_counts'],
                  'outputs': ['legacy_benchmark.md','legacy_benchmark.json','legacy_source_manifest.json']}, indent=2))
