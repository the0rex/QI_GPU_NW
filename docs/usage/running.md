# Running QI-ALIGN

Basic command:

qi-align --ref human.fa --qry chimp.fa --out results/

Custom config:

qi-align --ref A.fa --qry B.fa --config config/human-chimp.yaml --out output/

Outputs:
- alignment.cigar
- stats.json
- logs (stdout)
