import time
from abc import ABC, abstractmethod
import torch
import pathfinder_pcpage_functions as pf_pcpage
from collections import OrderedDict

class MLPrefetchModel_pathfinder_pcpage(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''

    @abstractmethod
    def load(self, path):
        '''
        Loads your model from the filepath path
        '''
        pass

    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        pass

    @abstractmethod
    def generate(self, data):
        print(torch.cuda.is_available())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name())
        print(torch.cuda.get_device_capability())
        print(len(data))
        '''
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        '''

        # constants
        pattern_length = 3
        confidence_threshold = 1
        min_confidence = -2
        correct_predictions = 0
        correct_offset_predictions = 0
        correct_delta_predictions = 0
        total_predictions = 0
        total_delta_prediction = 0
        total_offset_prediction = 0
        label_removal_counter = 0
        pc_removal_counter = 0
        page_removal_counter = 0

        # constants for snn
        delta_range_length = 127
        neuron_numbers = 50

        timestamps = 32
        # input_intensity = 10000
        input_intensity = 1576

        sanity_check_index = 3000
        refresh_index = 5000
        last_addr_index = 1000000

        # create snn
        single_network = pf_pcpage.CreateNetwork(pattern_length, confidence_threshold, min_confidence, delta_range_length, neuron_numbers, timestamps, input_intensity)

        prefetch_addresses = []
        # key = pc, value = dict_inner {key = page, value = [[delta pattern(length = 3)], last offset, neuron, offset_pre/delta_pre]}
        training_table = OrderedDict()

        cur_index = 0

        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            # print("---------------------------------")
            # print("current index: ", cur_index)
            # time.sleep(5)

            if cur_index % sanity_check_index == 0 and cur_index != 0:
                print("----- the current index: ", cur_index)
                print("correct prediction: ", correct_predictions)
                print("total predictions: ", total_predictions)
                print("correct/total predict: ", correct_predictions / total_predictions)
                print(" correct / total load instr: ", correct_predictions / cur_index)
                print(" ")
            if cur_index > last_addr_index:
                break
            if cur_index % refresh_index == 0 and cur_index != 0:
                single_network = pf_pcpage.CreateNetwork(pattern_length, confidence_threshold, min_confidence, delta_range_length, neuron_numbers, timestamps, input_intensity)

            cur_index += 1
            pc, page, offset = load_ip, load_addr >> 12, ((load_addr >> 6) & 0x3f)

            is_correct, label_removal_counter = pf_pcpage.check_hit(training_table, single_network.prediction_table, pc, page, offset, label_removal_counter)
            if is_correct:
                correct_predictions += 1
                if training_table[pc][page][3] == 1:
                    correct_offset_predictions += 1
                else:
                    correct_delta_predictions += 1

            single_network.label_unlabeled(pc, page, offset, training_table)

            # key = pc, value = inner_dict {key = page, value = [[delta pattern(length = 3)], last offset, neuron, offset_pre/delta_pre]}
            offset_prediction = 0
            # if pc in training_table:
            #     inner_dict = training_table[pc]
            if pc in training_table:
                inner_dict = training_table[pc]
                if page in inner_dict:
                    delta_pattern = pf_pcpage.calculate_deltas(inner_dict[page], offset)
                else:
                    delta_pattern = [offset, 0, 0]
                    offset_prediction = 1

            else:
                delta_pattern = [offset, 0, 0]
                offset_prediction = 1
                # training_table[pc] = {page: (delta_pattern, offset, -1)}
            # ----------------------has some problem

            output_neurons, predicted_deltas = single_network.make_prediction(delta_pattern)

            if predicted_deltas is not None:
                # todo why no error before?
                for predicted_delta in predicted_deltas:
                    if offset + predicted_delta > 0:
                        if offset_prediction == 1:
                            total_offset_prediction += 1
                        else:
                            total_delta_prediction += 1

                        total_predictions += 1
                        predicted_address = (int(page) << 12) + (int(offset + predicted_delta) << 6)
                        prefetch_addresses.append((instr_id, predicted_address))

            # if output_neurons:
            pc_removal_counter, page_removal_counter = pf_pcpage.update_training_table(pc, page, offset, output_neurons, delta_pattern, training_table, offset_prediction, pc_removal_counter, page_removal_counter)
            # print("check training table", training_table)

        print("correct prediction: ", correct_predictions)
        print("total predictions: ", total_predictions)
        print("correct/total predict: ", correct_predictions / total_predictions)
        print(" correct / total load instr: ", correct_predictions / cur_index)

        print("correct offset prediction: ", correct_offset_predictions)
        print("total offset prediction: ", total_offset_prediction)
        # print("correct offset prediction/total offset prediction:", correct_offset_predictions/total_offset_prediction)

        print("correct delta prediction: ", correct_delta_predictions)
        print("total delta prediction: ", total_delta_prediction)
        print("correct delta prediction/total delta prediction:", correct_delta_predictions / total_delta_prediction)

        print("# of prefetches ", len(prefetch_addresses))

        return prefetch_addresses
# Replace this if you create your own model
Model = MLPrefetchModel_pathfinder_pcpage
