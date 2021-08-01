import os
import unicodedata
from service_vncorenlp.custom_vncorenlp import VnCoreNLP
import itertools
import subprocess
import time

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

class AutoPhraseVN():
    def __init__(self):
        ## VncoreNLP
        self.vncore = VnCoreNLP()

        ## Constants
        self.OOV_ID = "-1111"
        self.OOV_WORD = "<OOV>"

        ## Các file model và từ điển
        self.MODEL_FOLDER = "models/VI"
        self.TOKEN_MAPPING = os.path.join(ROOT_PATH, self.MODEL_FOLDER, "token_mapping.txt")
        self.MODEL_PATH = os.path.join(ROOT_PATH, self.MODEL_FOLDER, "segmentation.model")
        self.PUNCTIONS_PATH = os.path.join(ROOT_PATH,\
                                         "tools/tokenizer/resources/indo_european_punctuation_mapping.txt")

        ## Các file tmp
        self.POS_TAG_FILE = os.path.join(ROOT_PATH, "tmp/pos_tags_tokenized_text_to_seg.txt")
        self.RAW_TOKENIZED_FILE = os.path.join(ROOT_PATH, "tmp/raw_tokenized_text_to_seg.txt")
        self.TOKENIZED_FILE = os.path.join(ROOT_PATH, "tmp/tokenized_text_to_seg.txt")
        self.SEGMENTED_SENTENCES_FILE = os.path.join(ROOT_PATH, "tmp/tokenized_segmented_sentences.txt")

        #### Load Vocab ####
        self.load_puctions()
        self.load_vocab()

    def segment(self, list_sentences, thread=8, threshold_mutil=0.7, threshold_single=0.9):
        """Phân đoạn một danh sách các câu đầu vào
        
        Parameters: 
            list_sentences (list): Mảng các câu đầu vào cần trích xuất cụm từ
            thread (int): số luồng chạy autophrase
            threshold_mutil(float): là giá trị nằm trong khoảng 0-1, 
                                    xác định ngưỡng xác suất để chấp nhận một cụm da từ là cụm từ tích cực
            threshold_single(float): là giá trị nằm trong khoảng 0-1, 
                                    xác định ngưỡng xác suất để chấp nhận một từ đơn là cụm từ tích cực

        Returns: 
            list: Danh sách các câu dạng:
                {'text': 'kỹ_năng thuyết_trình và làm_việc nhóm',
                 'pos': 'N V Cc V N',
                 'phrases': [
                    {'phrase': 'kỹ_năng thuyết_trình', 'position': [0, 2]},
                    {'phrase': 'làm_việc nhóm', 'position': [3, 5]}]
                }
        """
        ## Tokenize
        self.token_sents, self.pos_tag_sents, self.token_id_sents = self.tokenize(list_sentences)
        self.save_tokened_file()

        ## Chạy segmentation
        self.run_phrase_segment(thread, threshold_mutil, threshold_single)

        ## Đọc kết quả và trả về
        accepted_pos = ["N", "V"]
        accepted_pos = []
        outputs = self.get_output(accepted_pos)
        return outputs

    def segment_large_data(self, list_sentences, language="VI", threshold_mutil=0.7, threshold_single=0.9):
        """ 
        Dùng AutoPhrase để trích xuất các cụm từ cho một lượng lớn câu đầu vào

        Parameters: 
            list_sentences (list): Mảng các câu cần trích xuất cụm từ, lưu ý với số lượng câu lớn hơn 1000 thì tốc độ sẽ nhanh hơn đáng kể so với dùng hàm segment()
            language (str): Ngôn ngữ của dữ liệu: Tiếng việt (VI), Tiếng Anh (EN),....
            threshold_mutil(float): là giá trị nằm trong khoảng 0-1, 
                                    xác định ngưỡng xác suất để chấp nhận một cụm da từ là cụm từ tích cực
            threshold_single(float): là giá trị nằm trong khoảng 0-1, 
                                    xác định ngưỡng xác suất để chấp nhận một từ đơn là cụm từ tích cực

        Returns: 
            bool: True - Nếu train model thành công, False - Nếu train model không thành công
        """
        tmp_input_file = os.path.join(ROOT_PATH, "tmp/input.txt")
        tmp_output_file = os.path.join(ROOT_PATH, "tmp/output.txt")

        list_sentences = list(map(lambda text: nomarlize_text(text), list_sentences))

        with open(tmp_input_file, "w") as file:
            file.write("\n".join(list_sentences))

        self.segment_from_file(tmp_input_file,tmp_output_file,language,threshold_mutil,threshold_single)

        with open(tmp_output_file, "r") as file:
            output = file.read()
            output = output.splitlines()
        list_output = []
        for i,sent in enumerate(output):
            sent = sent.replace("_"," ")
            text = list_sentences[i]
            sentence = {
                    "text": text,
                    "phrases": []
                }
            start = sent.find("<phrase>")
            while start != -1 :
                sent = sent[:start] + sent[start+8:]
                end = sent.find("</phrase>")
                sent = sent[:end] + sent[end+9:]
                sentence["phrases"].append({
                    "phrase": text[start:end],
                    "position": [start,end]
                })
                start = sent.find("<phrase>")

            list_output.append(sentence)
        return list_output

    @staticmethod
    def segment_from_file(input_file, output_file, language="VI", threshold_mutil=0.6, threshold_single=0.9):
        """ 
        Dùng AutoPhrase để trích xuất các cụm từ cho các câu trong file input.

        Parameters: 
            input_file (str): Đường dẫn đến file chứa dữ liệu cần trích xuất các cụm từ
            output_file (str): Đường dẫn đến file chứa dữ liệu sau khi đã trích xuất cụm từ.
            language (str): Ngôn ngữ của dữ liệu: Tiếng việt (VI), Tiếng Anh (EN),....
            threshold_mutil(float): là giá trị nằm trong khoảng 0-1, 
                                    xác định ngưỡng xác suất để chấp nhận một cụm da từ là cụm từ tích cực
            threshold_single(float): là giá trị nằm trong khoảng 0-1, 
                                    xác định ngưỡng xác suất để chấp nhận một từ đơn là cụm từ tích cực

        Returns: 
            bool: True - Nếu train model thành công, False - Nếu train model không thành công
        """
        input_file = os.path.realpath(input_file)
        output_file = os.path.realpath(output_file)
        cmd = f"""cd {ROOT_PATH}; \
        ./phrasal_segmentation.sh {input_file} {output_file} {language} {threshold_mutil} {threshold_single}"""
        return_code = subprocess.call(cmd, shell=True)
        if return_code == 0:
            return True
        else:
            return False

    @staticmethod
    def train(file_raw_data):
        """ 
        Đào tạo model cho AutoPhrase.

        Parameters: 
            file_raw_data (str): Đường dẫn đến file chứa dữ liệu dùng để train model
                                    Đường dẫn tương đối hoặc tuyệt đối
        Returns: 
            bool: True - Nếu train model thành công, False - Nếu train model không thành công
        """
        file_raw_data = os.path.realpath(file_raw_data)
        cmd = f"""cd {ROOT_PATH}; \
        ./train_auto_phrase.sh {file_raw_data} """

        return_code = subprocess.call(cmd, shell=True)
        if return_code == 0:
            return True
        else:
            return False

    def demo(self, input_file, output_file="demo.html", language="VI", threshold_mutil=0.7, threshold_single=0.9):
        """Demo trích xuất cụm từ
        Đầu vào sẽ là một file text chứa các câu cần trích xuất cụm từ
        Đầu ra sẽ là một file html chứa các câu đầu vào, và bôi đậm các cụm từ trong đó
        """
        def convert_output2html(doc):
            list_token_phrase = []
            start = 0
            list_tokens = doc["text"].split()
            for phrase in doc["phrases"]:
                end = phrase[1][0]
                list_token_phrase.extend(list_tokens[start:end])
                list_token_phrase.append(f"<b>\"{phrase[0]}\"</b>")
                start = phrase[1][1]
            list_token_phrase.extend(list_tokens[start:len(list_tokens)])
            return " ".join(list_token_phrase)

        input_file = os.path.realpath(input_file)
        output_file = os.path.realpath(output_file)
        with open(input_file, "r") as file:
            inputs = file.read()
            inputs = inputs.splitlines()

        ## Tokenize
        self.token_sents, self.pos_tag_sents, self.token_id_sents = self.tokenize(inputs)
        self.save_tokened_file()

        ## Chạy segmentation
        self.run_phrase_segment(8, threshold_mutil, threshold_single)

        ## Đọc kết quả và trả về
        accepted_pos = []
        outputs = self.get_output(accepted_pos)

        with open('./label_final_v9.txt', 'w') as filehandle:
            for listitem in outputs:
                for p in listitem['phrases']:
                    filehandle.write('%s\n' % p[0])

        outputs = list(map(lambda doc: convert_output2html(doc), outputs))
        with open(output_file, "w") as file:
            file.write("<br>\n".join(outputs))
        return

    def demo_old(self, input_file, output_file="output.html", language="VI", threshold_mutil=0.6, threshold_single=0.9):
        """Demo trích xuất cụm từ
        Đầu vào sẽ là một file text chứa các câu cần trích xuất cụm từ
        Đầu ra sẽ là một file html chứa các câu đầu vào, và bôi đậm các cụm từ trong đó
        """
        input_file = os.path.realpath(input_file)
        output_file = os.path.realpath(output_file)
        tmp_output_file = os.path.join(ROOT_PATH, "tmp/output.txt")

        self.segment_from_file(input_file,tmp_output_file,language,threshold_mutil,threshold_single)

        with open(tmp_output_file, "r") as file:
            sentences = file.read()
        sentences = sentences.replace("<phrase>", "\"<b>")
        sentences = sentences.replace("</phrase>", "</b>\"")
        sentences = sentences.replace("\n", "<br>")
        with open(output_file, "w") as file:
            file.write(sentences)
        return

    def run_phrase_segment(self, THREAD, HIGHLIGHT_MULTI, HIGHLIGHT_SINGLE):
        cmd = f"""cd {ROOT_PATH}; \
        ./bin/segphrase_segment \
        --pos_tag \
        --thread {THREAD} \
        --model {self.MODEL_PATH} \
        --highlight-multi {HIGHLIGHT_MULTI} \
        --highlight-single {HIGHLIGHT_SINGLE}"""

        subprocess.call(cmd, shell=True)
        return

    def load_puctions(self):
        """Load các dấu câu
        """
        with open(self.PUNCTIONS_PATH, "r") as file:
            punctions = file.read()
            punctions = punctions.splitlines()

        self.PUNCTIONS_MAP = {}
        self.PUNCTIONS = set()
        for line in punctions:
            key, value = line.split("\t")
            self.PUNCTIONS_MAP[key] = value
            self.PUNCTIONS.add(value)
        return


    def load_vocab(self):
        """Load từ điển trong file token_mapping.txt
        """
        with open(self.TOKEN_MAPPING, "r") as file:
            token_map = file.read()
            token_map = token_map.splitlines()

        self.idx2word = {}
        self.word2idx = {}
        for line in token_map:
            key, word = line.split("\t")
            self.idx2word[key] = word
            self.word2idx[word] = key

        ## Thêm hai dấu hiệu phân tách cụm từ
        self.idx2word["<phrase>"] = "<phrase>"
        self.idx2word["</phrase>"] = "</phrase>"
        self.idx2word[self.OOV_ID] = self.OOV_WORD

        ## Puctions
        for key, value in self.PUNCTIONS_MAP.items():
            self.word2idx[key] = value

        return

    def tokenize(self, list_sentences):
        list_tmp = map(lambda sent: self.token_sentence(sent), list_sentences)
        token_sents, pos_tag_sents, token_id_sents = list(zip(*list_tmp))
        return token_sents, pos_tag_sents, token_id_sents

    def token_sentence(self, sentence):
        tokens, pos_tags = self.tokenize_sentence_by_vncore(sentence)
        token_ids = list(map(lambda t: self.get_word_id(t), tokens))
        return (" ".join(tokens), " ".join(pos_tags), " ".join(token_ids))

    def tokenize_sentence_by_vncore(self, sentence):
        """Hàm tách từ và gán thẻ pos sử dụng VNcoreNLP
        """
        tokens = self.vncore.pos_tag(sentence)
        tokens = flatten_list(tokens)

        tokens, pos_tags = list(zip(*tokens))
        return tokens, pos_tags


    def convert_idx2sentence(self, sent):
        """
        Chuyển một chuỗi idx thành văn bản, ví dụ:
                <phrase> 428 430 </phrase> 9 <phrase> 20 344 </phrase>
        ===>    <phrase> kỹ_năng thuyết_trình </phrase> và <phrase> làm_việc nhóm </phrase>
        """
        return " ".join(map(lambda l: self.idx2word[l], sent.split()))

    def get_word_form_id(self, idx):
        """
        Trả về một từ tương ứng với idx
        """
        return self.idx2word[idx]

    def get_word_id(self, word):
        """
        Trả về id của một từ được lưu trong file token_mapping.txt
        Nếu từ không tồn tại trong file, sẽ trả về OOV_ID
        """
        try:
            return self.word2idx[word.lower()]
        except:
            return self.OOV_ID

    def save_tokened_file(self):
        """Lưu các dữ liệu đã tokenized và pos_tagged vào file tạm
        """
        with open(self.POS_TAG_FILE, "w") as file:
            file.write("\n".join(self.pos_tag_sents))

        with open(self.RAW_TOKENIZED_FILE, "w") as file:
            file.write("\n".join(self.token_sents))

        with open(self.TOKENIZED_FILE, "w") as file:
            file.write("\n".join(self.token_id_sents))
        return

    def get_output(self, accepted_pos):
        """Hàm đọc kết quả phân đoạn từ file tmp/tokenized_segmented_sentences.txt và trả về kết quả
        Parameters:
            accepted_pos (list): Các từ đơn có thẻ Pos nằm trong accepted_pos sẽ được trả về cùng các cụm từ khác
        Returns:
            list: Danh sách các câu dạng:
                {
                    "text": Câu đầu vào đã được tách từ,
                    "pos": Chuỗi thẻ pos tương ứng của câu đầu vào,
                    "phrases": [Danh sách các cụm từ trong câu]
                }
        """
        with open(self.SEGMENTED_SENTENCES_FILE, "r") as file:
            segmentations = file.read()
            segmentations = segmentations.splitlines()
            segmentations = " ".join(segmentations).split()

        list_output = []
        seg_idx = 0
        len_chain_segmentation = len(segmentations)
        for sent_id, sentence in enumerate(self.token_id_sents):
            tmp = {
                "text": self.token_sents[sent_id],
                "pos": self.pos_tag_sents[sent_id],
                "phrases": []
            }
            raw_sentence = self.token_sents[sent_id].split()
            pos_sentence = self.pos_tag_sents[sent_id].split()
            sentence = sentence.split()

            in_phrase = False
            for i, word_id in enumerate(sentence):

                if word_id in self.PUNCTIONS:
                    if in_phrase and segmentations[seg_idx] != "</phrase>":
                        end += 1
                    continue

                if in_phrase:
                    end += 1

                if segmentations[seg_idx] == "<phrase>":
                    seg_idx += 1
                    in_phrase = True
                    start = i
                    end = start
                elif segmentations[seg_idx] == "</phrase>":
                    seg_idx += 1
                    in_phrase = False
                    tmp["phrases"].append((" ".join(raw_sentence[start:end]), (start, end)))
                    if segmentations[seg_idx] == "<phrase>":
                        seg_idx += 1
                        in_phrase = True
                        start = i
                        end = start
                if segmentations[seg_idx] == word_id:
                    seg_idx += 1
                    ## Lấy thêm các từ có thẻ pos nằm trong accepted_pos
                    if pos_sentence[i][0] in accepted_pos and not in_phrase:
                        tmp["phrases"].append((raw_sentence[i],(i,i+1)))
                    continue
                else:
                    print(f"Câu thứ {sent_id}:  {seg_idx} - {segmentations[seg_idx]}  {word_id}")

            if seg_idx < len_chain_segmentation and segmentations[seg_idx] == "</phrase>":
                seg_idx += 1
                end += 1
                in_phrase = False
                tmp["phrases"].append((" ".join(raw_sentence[start:end]), (start, end)))
            list_output.append(tmp)
        return list_output

def flatten_list(l):
    """Làm phẳng một danh sách hai chiều không đồng nhất, ví dụ:
    [[1,2,3,4], [1,3], [5], [6,7,8]] --> [1, 2, 3, 4, 1, 3, 5, 6, 7, 8]
    """
    return list(itertools.chain.from_iterable(l))

def nomarlize_text(string):
    replace_list = {
        'ả': 'ả', 'ố': 'ố', 'u´': 'ố','ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
        'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề','ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
        'ẻ': 'ẻ', 'àk': u' à ','aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ','ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á'}
    for k, v in replace_list.items():
        string = string.replace(k, v)

    string = normalize_unicode(string)
    string = string.replace("_"," ").replace(""," ")
    string = " ".join(string.split())
    return string

def normalize_unicode(text):
    """Hàm chuẩn hóa unicode
    """
    return unicodedata.normalize('NFC', text)


if __name__ == "__main__":
    autophrase = AutoPhraseVN()
    autophrase.demo('./test_dantri.txt')
    # label_truth = []
    # fo = open("label_dantri.txt", "r")
    # a = fo.readlines()
    # label_truth.extend(a)
    # output_truth = []
    # for l in label_truth:
    #     l = l.strip('\n')
    #     out_tokens, out_pos = autophrase.tokenize_sentence_by_vncore(l)
    #     if not out_tokens:
    #         continue
    #     output_truth.append(" ".join(out_tokens))
    #
    # with open('label_dantri_final.txt', 'w') as filehandle:
    #     for listitem in output_truth:
    #         listitem = str(listitem)
    #         filehandle.write('%s\n' % listitem)