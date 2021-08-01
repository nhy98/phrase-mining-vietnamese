# Automatically Phrase Mining in Vietnamese Documents

## Requirements

Linux or MacOS with g++ and Java installed.

Ubuntu:

* g++ 4.8 `$ sudo apt-get install g++-4.8`
* Java 8 `$ sudo apt-get install openjdk-8-jdk`
* curl `$ sudo apt-get install curl`

MacOS:

*   g++ 6 `$ brew install gcc6`
*   Java 8 `$ brew update; brew tap caskroom/cask; brew install Caskroom/cask/java`


#### Phrase Mining Step

### Training
```
$ ./test.sh
```

The default run will download an Vietnamese corpus data and run AutoPhrase to get 3 ranked lists of phrases as well as 2 segmentation model files under the
```MODEL``` (i.e., ```models/VI```) directory. 
* ```AutoPhrase.txt```: the unified ranked list for both single-word phrases and multi-word phrases. 
* ```AutoPhrase_multi-words.txt```: the sub-ranked list for multi-word phrases only. 
* ```AutoPhrase_single-word.txt```: the sub-ranked list for single-word phrases only.
* ```segmentation.model```: AutoPhrase's segmentation model (saved for later use).
* ```token_mapping.txt```: the token mapping file for the tokenizer (saved for later use).


### Testing

Using model saved in folder models/VI to test on testing dataset related to medical and health.

Run server VnCoreNLP
```
$ python start.py
```
Run demo on testing file: test_dantri.txt

```
$ python demo.py
```



