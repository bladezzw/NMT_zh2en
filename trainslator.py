# import the necessary packages

from models import EncoderRNN, LuongAttnDecoderRNN
from utils import *

if __name__ == '__main__':
    input_lang = Lang('data/WORDMAP_zh.json')
    output_lang = Lang('data/WORDMAP_en.json')
    print("input_lang.n_words: " + str(input_lang.n_words))
    print("output_lang.n_words: " + str(output_lang.n_words))

    checkpoint = '{}/BEST_checkpoint.tar'.format(save_dir)  # model checkpoint
    print('checkpoint: ' + str(checkpoint))
    # Load model
    checkpoint = torch.load(checkpoint)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']

    print('Building encoder and decoder ...')
    # Initialize encoder & decoder models
    encoder = EncoderRNN(input_lang.n_words, hidden_size, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, decoder_n_layers, dropout)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module

    searcher = GreedySearchDecoder(encoder, decoder)
    while True:
        input_ = input("请输入中文: (输入q退出)\n")
        if input_ == 'q':
            break
        sentence_zh = input_.strip()
        seg_list = jieba.cut(sentence_zh)
        sample = encode_text(input_lang.word2index, list(seg_list))

        input_sentence = ''.join([input_lang.index2word[token] for token in sample if token != EOS_token])

        decoded_words = evaluate(searcher, input_sentence, input_lang, output_lang)
        print('译文 {}'.format(' '.join(decoded_words)))