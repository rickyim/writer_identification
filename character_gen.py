import numpy as np
import os
#directory of the dataset
#dir_train = '../WriterID/Data/Train'
dir_test = '../WriterID/Data/Validation'
#write directory
write_test = './character_test'
if not os.path.isdir(write_test):
    os.mkdir(write_test)

#length of the sequence
seq_length = 40
interval = 20
num_sample_test = 5
#
filenames = os.listdir(dir_test)
numclass = len(filenames)
#testing
for i in range(numclass):
    #characters belong to a person
    character_person = np.load(os.path.join(dir_test, filenames[i]))
    #a character
    #print(' {} th person has {} characters'.format(i, len(character_person)))
    for character_id in range(len(character_person)):
        datapoints = np.zeros((1, 3))
        character = character_person[character_id]

        for stroke_id in range(len(character)):
            # a stroke
            stroke = character[stroke_id]
            stroke_len = len(stroke)
            for point_id in range(stroke_len):
                datapoints = np.append(datapoints, np.expand_dims(np.append(stroke[point_id], 1), axis = 0), axis=0)
            datapoints[-1, -1] = 0

        x_min = np.min(datapoints[1:, 0])
        x_max = np.max(datapoints[1:, 0])
        y_min = np.min(datapoints[1:, 1])
        y_max = np.max(datapoints[1:, 1])        
        datapoints[1:, 0] = (datapoints[1:, 0] - x_min)/(x_max - x_min)
        datapoints[1:, 1] = (datapoints[1:, 1] - y_min)/(y_max - y_min)


        points_seq = np.array([np.append((datapoints[i+1]-datapoints[i])[0:2], datapoints[i, 2]) for i in range(1, len(datapoints)-1)])
        points_length = len(points_seq)
        
        idx = 0
        print(' {} th person {} th character has {} points'.format(i, character_id, points_length))
        while(points_length>0):
            # a sequence
            if(points_length <= seq_length):
                save_seq = np.zeros((seq_length,3))
                save_seq[-points_length:] = np.array(points_seq[-points_length:])
            else:
                save_seq = np.array(points_seq[-points_length:-points_length+seq_length])
            points_length = points_length - interval
            scalar_label = np.zeros((1,))
            scalar_label[0] = i
            instance = {}
            instance['data'] = save_seq
            instance['gt'] = scalar_label
            np.save(os.path.join(write_test, str(seq_length)+'_'+str(i)+'-'+str(character_id)+'-'+str(idx)+'.npy'), instance)
            idx = idx + 1
            if idx>num_sample_test:
                break
        print('the {} th person {} th character has {} testing stroke sequences'.format(i+1, character_id+1, idx+1))
    
