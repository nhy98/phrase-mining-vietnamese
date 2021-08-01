#!/bin/bash
DATA_DIR=${DATA_DIR:- data}
# RAW_TRAIN là dữ liệu đầu vào của AutoPhrase với mỗi dòng là một tài liệu (document)
RAW_TRAIN=${DATA_DIR}/VI/final_v2.txt
# Khi FIRST_RUN được đặt là 1 thì AutoPhrases sẽ chạy tất cả các bước tiền xủ lý dữ liệu (tách từ,gán pos,...) 
# Nếu không thì autoPhrase sẽ chạy trực tiếp từ dữ liệu đã được xử lý như trong thư mục tmp/ .
FIRST_RUN=${FIRST_RUN:- 1}
# Khi ENABLE_POS_TAGGING được đặt là 1 thì AutoPhrases sẽ sử dụng tính năng Pos tagging trong khai thác cụm từ. 
# tức là chức năng Pos-guide phrasal segment được kích hoạt
# Nếu không thì AutoPhrases sẽ sử dụng hình phạt đơn giản cho cụm từ dài như trong SegPhrase.
ENABLE_POS_TAGGING=${ENABLE_POS_TAGGING:- 1}
# MIN_SUP là tần suất xuất hiện tối thiểu của 1 cụm n_gram để trở thành một cadidate phrases
MIN_SUP=${MIN_SUP:- 20}
# Chỉ định bao nhiêu luồng được sử dụng cho AutoPhrases
THREAD=${THREAD:- 6}

### Begin: tham số được đề xuất ###
MAX_POSITIVES=-1
LABEL_METHOD=DPDN

### Tập nhãn do chuyên gia gán ???? chưa hiểu cấu trúc dữ liệu trong file này như thế nào
RAW_LABEL_FILE=${RAW_LABEL_FILE:-""}
### End: tham số được đề xuất ###

green=`tput setaf 2`
reset=`tput sgr0`
COMPILE=${COMPILE:- 1}

if [ $COMPILE -eq 1 ]; then
    echo ${green}===Compilation===${reset}
    bash compile.sh
fi
### Tạo thư mục chứa tạm các kết quả trung gian
mkdir -p tmp

############## Bắt đầu quá trình tách từ ##############
echo ${green}===Tokenization===${reset}

## đường dẫn đến file chạy tách từ
TOKENIZER="-cp .:tools/tokenizer/lib/*:tools/tokenizer/resources/:tools/tokenizer/build/ Tokenizer"
TOKENIZED_TRAIN=tmp/tokenized_train.txt
CASE=tmp/case_tokenized_train.txt
TOKEN_MAPPING=tmp/token_mapping.txt

### Nếu chạy lần đầu (FIRST RUN = 1) thì sẽ tách từ các câu ở dữ liệu đầu vào và cho vào tập Train
if [ $FIRST_RUN -eq 1 ]; then
    echo -ne "Current step: Tokenizing input file...\033[0K\r"
    time java $TOKENIZER -m train -i $RAW_TRAIN -o $TOKENIZED_TRAIN -t $TOKEN_MAPPING -c N -thread $THREAD
fi
### detect ngôn ngữ
LANGUAGE=`cat tmp/language.txt`
echo -ne "Detected Language: $LANGUAGE\033[0K\n"
TOKENIZED_STOPWORDS=tmp/tokenized_stopwords.txt
TOKENIZED_ALL=tmp/tokenized_all.txt
TOKENIZED_QUALITY=tmp/tokenized_quality.txt

## Tạo model cho từng ngôn ngữ
model_lang=models/$LANGUAGE
MODEL=${MODEL:- $model_lang}
## Tạo thư mục chứa model sau khi đã train xong
mkdir -p ${MODEL}

## load stopword
STOPWORDS=data/$LANGUAGE/stopwords.txt
## load wiki_all
ALL_WIKI_ENTITIES=data/$LANGUAGE/wiki_all.txt
## load wiki_quality
QUALITY_WIKI_ENTITIES=data/$LANGUAGE/wiki_quality.txt
LABEL_FILE=tmp/labels.txt
if [ $FIRST_RUN -eq 1 ]; then
    echo -ne "Current step: Tokenizing stopword file...\033[0K\r"
    ## tách từ cho các từ stopwords, ở đây xem stopword là các từ negative
    java $TOKENIZER -m test -i $STOPWORDS -o $TOKENIZED_STOPWORDS -t $TOKEN_MAPPING -l $LANGUAGE -c N -thread $THREAD
    echo -ne "Current step: Tokenizing wikipedia phrases...\033[0K\n"
    ## tách từ cho các cụm từ chất lượng trong wiki_quality.txt và wiki_all.txt
    java $TOKENIZER -m test -i $ALL_WIKI_ENTITIES -o $TOKENIZED_ALL -t $TOKEN_MAPPING -l $LANGUAGE -c N -thread $THREAD
    java $TOKENIZER -m test -i $QUALITY_WIKI_ENTITIES -o $TOKENIZED_QUALITY -t $TOKEN_MAPPING -l $LANGUAGE -c N -thread $THREAD
fi  
### END Tokenization ### Kết thúc quá trình tách từ ################

########## Bắt đầu quá trình gán nhãn thẻ Pos cho các từ ##############
if [[ $RAW_LABEL_FILE = *[!\ ]* ]]; then
	echo -ne "Current step: Tokenizing expert labels...\033[0K\n"
	java $TOKENIZER -m test -i $RAW_LABEL_FILE -o $LABEL_FILE -t $TOKEN_MAPPING -l $LANGUAGE -c N -thread $THREAD
else
	echo -ne "No provided expert labels.\033[0K\n"
fi

echo ${green}===Part-Of-Speech Tagging===${reset}

### Nếu chạy lần đầu (FIRST RUN = 1) thì sẽ gãn nhãn từ loại cho tập train (raw_tokenized_train.txt)
# với các ngôn ngữ JA, CN và OTHER thì sẽ không dùng tính năng Pos-guide phrasal segment, thay vào đó dùng hình phạt cho các 
# cụm từ dài như trong SegPhrase
if [ ! $LANGUAGE == "VI" ] && [ ! $LANGUAGE == "JA" ] && [ ! $LANGUAGE == "CN" ] && [ ! $LANGUAGE == "OTHER" ]  && [ $ENABLE_POS_TAGGING -eq 1 ] && [ $FIRST_RUN -eq 1 ]; then
    RAW=tmp/raw_tokenized_train.txt
    export THREAD LANGUAGE RAW
    bash ./tools/treetagger/pos_tag.sh
    mv tmp/pos_tags.txt tmp/pos_tags_tokenized_train.txt
fi

### END Part-Of-Speech Tagging ### Kết thúc quá trình gán nhãn thẻ Pos ##############


# ############## Bắt đầu quá trình trích xuất tự động các cụm từ ###############
echo ${green}===AutoPhrasing===${reset}

# nếu ENABLE_POS_TAGGING = 1 thì chạy file nhị phân ./bin/segphrase_train , có truyền vào pos_tag và pos_prune
if [ $ENABLE_POS_TAGGING -eq 1 ]; then
    time ./bin/segphrase_train \
        --pos_tag \
        --thread $THREAD \
        --pos_prune data/BAD_POS_TAGS.txt \
        --label_method $LABEL_METHOD \
		--label $LABEL_FILE \
        --max_positives $MAX_POSITIVES \
        --min_sup $MIN_SUP
else
    time ./bin/segphrase_train \
        --thread $THREAD \
        --label_method $LABEL_METHOD \
		--label $LABEL_FILE \
        --max_positives $MAX_POSITIVES \
        --min_sup $MIN_SUP
fi

### Lưu model phân đoạn câu lại
echo ${green}===Saving Model and Results===${reset}

cp tmp/segmentation.model ${MODEL}/segmentation.model
cp tmp/token_mapping.txt ${MODEL}/token_mapping.txt
cp tmp/language.txt ${MODEL}/language.txt

### END AutoPhrasing ### Kết thúc việc khai thác cụm từ tự động ###

### In các kết quả

echo ${green}===Generating Output===${reset}
java $TOKENIZER -m translate -i tmp/final_quality_multi-words.txt -o ${MODEL}/AutoPhrase_multi-words.txt -t $TOKEN_MAPPING -l $LANGUAGE -c N -thread $THREAD
java $TOKENIZER -m translate -i tmp/final_quality_unigrams.txt -o ${MODEL}/AutoPhrase_single-word.txt -t $TOKEN_MAPPING -l $LANGUAGE -c N -thread $THREAD
java $TOKENIZER -m translate -i tmp/final_quality_salient.txt -o ${MODEL}/AutoPhrase.txt -t $TOKEN_MAPPING -l $LANGUAGE -c N -thread $THREAD

# java $TOKENIZER -m translate -i tmp/distant_training_only_salient.txt -o results/DistantTraning.txt -t $TOKEN_MAPPING -c N -thread $THREAD

### END Generating Output for Checking Quality ###