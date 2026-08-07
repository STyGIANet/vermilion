DIR="./dump/"

cat $DIR/throughput-static.csv | grep 'mygrep' > $DIR/throughput.csv
cat $DIR/throughput-oblivious.csv | grep 'mygrep' | tail -n +2 >> $DIR/throughput.csv
cat $DIR/throughput-da-static.csv | grep 'mygrep' | tail -n +2 >> $DIR/throughput.csv
cat $DIR/throughput-da-periodic.csv | grep 'mygrep' | tail -n +2 >> $DIR/throughput.csv