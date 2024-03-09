# created on Oct 11

import sys
sys.path.insert(0, './bindsnet')

import torch

from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet import encoding
from collections import OrderedDict
import numpy as np

class CreateNetwork:

    def __init__(self, pattern_length, confidence_threshold, min_confidence, delta_range_length, neuron_numbers, timestamps, input_intensity):
        self.prediction_table = {}  # key = neuron, value = tuple(label, confidence)

        self.pattern_length = pattern_length
        self.confidence_threshold = confidence_threshold
        self.min_confidence = min_confidence
        self.delta_range_length = delta_range_length
        self.neuron_numbers = neuron_numbers
        self.timestamps = timestamps
        self.input_intensity = input_intensity

        self.network = DiehlAndCook2015(
            n_inpt=self.delta_range_length * self.pattern_length,
            n_neurons=self.neuron_numbers,
            exc=25.5,
            inh=20.5,
            # inh=4.5,
            # exc=25,
            # inh=6.5,
            dt=1,
            norm=(self.delta_range_length*self.pattern_length)/10,
            theta_plus=.05,
            inpt_shape=(1, self.delta_range_length * self.pattern_length),
        )

        # Simulation time.
        time = self.timestamps
        dt = 1
        device = 'cpu'

        # set up the spike monitors
        self.spikes = {}
        self.voltages = {}
        for layer in set(self.network.layers):
            self.spikes[layer] = Monitor(
                self.network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
            )
            self.network.add_monitor(self.spikes[layer], name="%s_spikes" % layer)


        self.output_monitor = self.spikes['Ae']
        self.voltages['Ae'] = Monitor(
            self.network.layers['Ae'], state_vars=["v"], time=int(time / dt), device=device
        )
        self.network.add_monitor(self.voltages['Ae'], name="%s_voltage" % layer)

    def feed_input(self, delta_pattern):

        current = torch.full([self.delta_range_length * self.pattern_length], 0)
        non_zero_indices = self.build_enlarged_input_array(delta_pattern)
        for index in non_zero_indices:
            current[index] = self.input_intensity

        # generate the spiking inputs for the network
        input_data_new = encoding.encodings.poisson(current, self.timestamps, 1, device='cpu')
        test_inputs = {"X": input_data_new}

        # pass the spike trains as input into the network
        self.network.run(inputs=test_inputs, time=self.timestamps)

        # array for keeping track of firing neurons
        output_neurons = []

        for time in range(self.timestamps):
            for i in range(self.neuron_numbers):
                if self.output_monitor.get('s')[time][0][i]:
                    # todo break the loops when we get enough output neurons
                    # spikecount[i] += 1
                    # it may have two same neurons fired in certain amount time
                    if i not in output_neurons:
                        output_neurons.append(i)

                    self.network.reset_state_variables()  # Reset state variables.
                    return output_neurons
        self.network.reset_state_variables()  # Reset state variables.
        return output_neurons

    # make the valid information more rich, make the valid pixel thicker
    def build_enlarged_input_array(self, delta_pattern):
        valid_index_list_1d = []
        for i in range(self.pattern_length):

            # TODO reordered input 0 1 2 3 4 5 > 4 1 5 3 0 2
            # version 11: use if, elif to reorder
            # if delta_pattern[i] % 6 == 0 and delta_pattern[i] + 4 <= 63:
            #     valid_index_1d = int(delta_pattern[i] + 4 + (self.delta_range_length - 1) / 2)
            # elif delta_pattern[i] % 6 == 2 and delta_pattern[i] + 3 <= 63:
            #     valid_index_1d = int(delta_pattern[i] + 3 + (self.delta_range_length - 1) / 2)
            # else:
                # version 10 code (remove indent to get back )
            valid_index_1d = int(delta_pattern[i] + (self.delta_range_length - 1) / 2)
            # print(valid_index_1d)
            valid_index_list_1d.append(valid_index_1d)

        input_array_prep = np.empty((0, 127), int)
        for valid_item_index in valid_index_list_1d:
            temp_list = np.zeros(127)
            temp_list = temp_list.astype(int)
            temp_list[valid_item_index] = 1
            input_array_prep = np.vstack((input_array_prep, temp_list))
        valid_index_list_2d = list(zip(*np.where(input_array_prep == 1)))

        row_bound = self.pattern_length - 1
        column_bound = self.delta_range_length - 1

        for index_2d in valid_index_list_2d:
            row_index = index_2d[0]
            column_index = index_2d[1]
            if row_index == row_bound:
                new_valid_row_index = row_index - 1
            else:
                new_valid_row_index = row_index + 1

                # prep_list[index_2d[0]-1, index_2d[1]] = 1
            if column_index == column_bound:
                new_valid_column_index = column_index - 1
            else:
                new_valid_column_index = column_index + 1

            input_array_prep[row_index, new_valid_column_index] = 1
            input_array_prep[new_valid_row_index, column_index] = 1
            input_array_prep[new_valid_row_index, new_valid_column_index] = 1

        input_array_prep = input_array_prep.flatten()
        non_zero_indices = np.where(input_array_prep == 1)[0]

        return non_zero_indices

    def make_prediction(self, delta_pattern):
        output_neurons = self.feed_input(delta_pattern)
        # tuple array for outputNeuron and delta
        prediction_deltas = []
        if output_neurons and output_neurons[0] in self.prediction_table:
            for delta_tuple in self.prediction_table[output_neurons[0]]:
                prediction_deltas.append( delta_tuple[0])
        else:
            # todo add nextline prefetcher here
            # prediction_delta += 1
            prediction_deltas = None


        # check prediction table if that's okay to generate prediction(if there is neuron label pair )
        # if not add neuron to training table
        # update training table
        # update prediction table, if there is existed neuron # in prediction table
        return output_neurons, prediction_deltas

    def label_unlabeled(self, pc, page, offset, training_table):
        if pc in training_table and page in training_table[pc]:
            fired_neuron = training_table[pc][page][2]
            delta = offset - training_table[pc][page][1]
            if fired_neuron >= 0 and fired_neuron not in self.prediction_table:
                # assign new label and confidence
                self.prediction_table[fired_neuron] = [(delta, 0)]
            # add the second label confidence pair
            elif fired_neuron >= 0 and fired_neuron in self.prediction_table:
                first_delta_tuple = self.prediction_table[fired_neuron][0]
                if len(self.prediction_table[fired_neuron]) == 1 and delta != first_delta_tuple[0]:
                    self.prediction_table[fired_neuron].append((delta, 0))


# use the current page to find firing neuron, and use neuron to find the label,
# comparing label with current offset, then update confidence in prediction table
def check_hit(training_table, prediction_table, pc, page, offset, label_removal_counter):
    # check hit doesn't need to check pc info
    # traning table:         # key = pc,

    # value = dict_inner {key = page, value = tuple([delta pattern(length = 3)], last offset, neuron, offset_pre/delta_pre)}
    if pc in training_table and page in training_table[pc]:

        fired_neuron = training_table[pc][page][2]

        if fired_neuron > 0 and fired_neuron in prediction_table:
            is_correct_bit = 0
            label_toRemove_idx = None
            for i in range(0, len(prediction_table[fired_neuron])):
                delta_tuple = prediction_table[fired_neuron][i]
                confidence = delta_tuple[1]
                if delta_tuple[0] == (offset - training_table[pc][page][1]):
                    if confidence < 5:
                        prediction_table[fired_neuron][i] = (delta_tuple[0], confidence + 1)
                        is_correct_bit = 1
                    break

                else:
                    if confidence < -2:
                        label_removal_counter += 1
                        label_toRemove_idx = i
                    else:
                        prediction_table[fired_neuron][i] = (delta_tuple[0], confidence - 1)
            if label_toRemove_idx is not None:
                del prediction_table[fired_neuron][label_toRemove_idx]
            return is_correct_bit, label_removal_counter
        else:
            return 0, label_removal_counter
    else:
        return 0, label_removal_counter


def calculate_deltas(inner_dict_values, offset):
    old_deltas = inner_dict_values[0]
    new_deltas = [old_deltas[1], old_deltas[2], (offset-inner_dict_values[1])]

    return new_deltas


def update_training_table(pc, page, offset, fired_neurons, new_delta_pattern, training_table, offset_prediction, pc_removal_counter, page_removal_counter):

    if len(training_table) > 8:
        training_table.popitem(last=False)
        pc_removal_counter += 1

    if pc not in training_table:
        training_table[pc] = OrderedDict()
    if len(training_table[pc]) > 128:
        training_table[pc].popitem(last=False)
        page_removal_counter += 1

    if fired_neurons:
        training_table[pc][page] = [new_delta_pattern, offset, fired_neurons[0], offset_prediction]
    else:
        training_table[pc][page] = [new_delta_pattern, offset, -1, offset_prediction]
    training_table.move_to_end(pc)
    training_table[pc].move_to_end(page)

    return pc_removal_counter, page_removal_counter


