# Download if not already existing

if [ ! -d $DATA/mini_libri/LibriSpeech/train-clean-5/ ]; then 
  mkdir $DATA/mini_libri/
  cd $DATA/mini_libri/
  wget http://www.openslr.org/resources/31/train-clean-5.tar.gz
  tar -xvzf train-clean-5.tar.gz
  rm train-clean-5.tar.gz
  cd -
fi

if [ ! -d $DATA/mini_libri/LibriSpeech/dev-clean-2/ ]; then 
  mkdir $DATA/mini_libri/
  cd $DATA/mini_libri/
  wget http://www.openslr.org/resources/31/dev-clean-2.tar.gz
  tar -xvzf dev-clean-2.tar.gz
  rm dev-clean-2.tar.gz
  cd -
fi

if [ ! -d $DATA/mini_libri/LibriSpeech/test-clean/ ]; then 
  mkdir $DATA/mini_libri/
  cd $DATA/mini_libri/
  wget http://www.openslr.org/resources/31/test-clean.tar.gz
  tar -xvzf test-clean.tar.gz
  rm test-clean.tar.gz
  cd -
fi


echo "manifesting"

mkdir -p data

python manifest.py $DATA/mini_libri/LibriSpeech/test-clean/ --dest tmp --valid-percent 0
echo $DATA/mini_libri/LibriSpeech/test-clean > data/test.tsv
tail -n +2 tmp/train.tsv | sort -k2,2nr >> data/test.tsv 

python manifest.py $DATA/mini_libri/LibriSpeech/dev-clean-2/ --dest tmp --valid-percent 0
echo $DATA/mini_libri/LibriSpeech/dev-clean-2 > data/valid.tsv
tail -n +2 tmp/train.tsv | sort -k2,2nr >> data/valid.tsv 

python manifest.py $DATA/mini_libri/LibriSpeech/train-clean-5/ --dest tmp --valid-percent 0
echo $DATA/mini_libri/LibriSpeech/train-clean-5 > data/train.tsv
tail -n +2 tmp/train.tsv | sort -k2,2nr >> data/train.tsv 

rm -r tmp

echo "writing reference trn"

folder=$DATA/mini_libri/LibriSpeech/dev-clean-2
tail -n +2 data/valid.tsv | cut -f1 | while read file; do
  read -r speaker chapter num < <(awk -F'[-/.]' '{print $1, $2, $5}' <<< "$file")
  line=$(grep -- "-${num} " $folder/${speaker}/${chapter}/${speaker}-${chapter}.trans.txt)
  utt_id=$(echo $line | cut -f1 -d " ")
  ref=$(echo $line | cut -f2- -d " ")
  echo "${ref} (${utt_id})" >> data/valid.trn
done 

folder=$DATA/mini_libri/LibriSpeech/test-clean
tail -n +2 data/test.tsv | cut -f1 | while read file; do
  read -r speaker chapter num < <(awk -F'[-/.]' '{print $1, $2, $5}' <<< "$file")
  line=$(grep -- "-${num} " $folder/${speaker}/${chapter}/${speaker}-${chapter}.trans.txt)
  utt_id=$(echo $line | cut -f1 -d " ")
  ref=$(echo $line | cut -f2- -d " ")
  echo "${ref} (${utt_id})" >> data/test.trn
done 
