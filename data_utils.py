from music_utils import * 
from preprocess import * 
from keras.utils import to_categorical

chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
N_tones = len(set(corpus))
n_a = 64
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def load_music_utils():
    chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)   
    return (X, Y, N_tones, indices_tones)


def generate_music(inference_model, corpus = corpus, abstract_grammars = abstract_grammars, tones = tones, tones_indices = tones_indices, indices_tones = indices_tones, T_y = 10, max_tries = 1000, diversity = 0.5):

    
    out_stream = stream.Stream()
    
    curr_offset = 0.0                                     
    num_chords = int(len(chords) / 3)                    
    
    print("Predicting new values for different set of chords.")

    for i in range(1, num_chords):
        
        curr_chords = stream.Voice()
        
        for j in chords[i]:

            curr_chords.insert((j.offset % 4), j)
        
        _, indices = predict_and_sample(inference_model)
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        
        predicted_tones +=  pred[-1]
                
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

        predicted_tones = prune_grammar(predicted_tones)
        
        sounds = unparse_grammar(predicted_tones, curr_chords)

        sounds = prune_notes(sounds)

        sounds = clean_up_notes(sounds)

        print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (len([k for k in sounds if isinstance(k, note.Note)]), i))
        
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("output/my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()
    
    
    return out_stream


def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):

    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    indices = np.argmax(pred, axis = -1)
    results = to_categorical(indices, num_classes=78)
    
    return results, indices