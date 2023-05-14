import json
import networkx as nx
import copy
import itertools
import queue
import numpy as np
from utils import *

ids = ["111111111", "222222222"]

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100

PASSENGERS_ON_TAXI_WEIGHT = 3.5
PASSENGERS_ON_MAP_WEIGHT = 7.2
DISTANCE_TO_DEST_WEIGHT = 1


class OptimalTaxiAgent:
    def __init__(self, initial):
        self.initial = json.dumps(initial)
        self.map = initial['map']
        self.gass_station = ()
        self.G = self._build_graph(initial['map'])
        self._short_distances, self._short_distances_paths = self._create_shortest_path_distances(self.G)
        self.MDP = {}
        self.iteration_stack = queue.LifoQueue()
        self.taxi_full_fuel = {taxi: initial['taxis'][taxi]['fuel'] for taxi in initial['taxis']}
        # Create MDP
        self.MDP_init(self.initial)
        discount_factor = 1
        self.value_iteration(discount_factor)
        self.our_policy()

    def _build_graph(self, game_map):
        """
        Builds the game graph where a node is represented as (x,y) coordinate, and 'I' is not reachable node
        :param game_map: the map of the game
        :return: graph representing the game map
        """
        G = nx.Graph()
        rows, cols = len(game_map), len(game_map[0])
        for i in range(rows):
            for j in range(cols):
                if game_map[i][j] == 'I':
                    continue
                if game_map[i][j] == 'G':
                    self.gass_station += ((i, j),)
                if i + 1 < rows and game_map[i + 1][j] != 'I':
                    G.add_edge((i, j), (i + 1, j))
                if i - 1 >= 0 and game_map[i - 1][j] != 'I':
                    G.add_edge((i, j), (i - 1, j))
                if j + 1 < cols and game_map[i][j + 1] != 'I':
                    G.add_edge((i, j), (i, j + 1))
                if j - 1 >= 0 and game_map[i][j - 1] != 'I':
                    G.add_edge((i, j), (i, j - 1))
        return G

    def _create_shortest_path_distances(self, G):
        """
        Creates shortest paths dictionary
        :param G: graph object
        :return: dictionary of shortest paths
        """
        paths_len, paths = {}, {}
        for n1 in G.nodes:
            for n2 in G.nodes:
                if n1 != n2:
                    path = nx.shortest_path(G, n1, n2)[1:]
                    paths[(n1, n2)] = path
                    paths_len[(n1, n2)] = len(path)
                else:
                    paths[(n1, n2)] = []
                    paths_len[(n1, n2)] = 0
        return paths_len, paths

    def dest_check(self, comb, taxi_c_loc, com):
        """
        Function that checks if the coordinates of the destination is legal
        """
        # (('move', 'taxi 1', (2, 3)), ('move', 'taxi 2', (0, 3)))
        for i, ac in enumerate(comb):
            loc_set = ()
            if ac[0] in com:
                temp = taxi_c_loc[i]
                x, y = temp[0], temp[1]
                loc_set += (x, y)
            if ac[0] == 'move' and ac[2] in loc_set:
                return False
            if ac[0] == 'move':
                loc_set += ac[2]
        return True

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""

        if type(state) == dict:
            state_dictionary = state.copy()
        else:
            state_dictionary = json.loads(state)

        if state_dictionary['turns to go'] == 0:
            return ['terminate']
        com = ('pick up', 'drop off', 'wait', 'refuel')
        taxis_places = []
        taxi_c_loc = []
        taxi_tup = state_dictionary["taxis"].items()
        actions_set = ()
        taxi_num = 0
        for i, t in enumerate(taxi_tup):
            taxis_places.append(())
            taxi_c_loc.append(t[1]['location'])
            taxi_num += 1
        for index, t in enumerate(taxi_tup):
            temp_texi = ()
            loc = t[1]["location"]
            x, y = loc[0], loc[1]
            options = [(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0] , loc[1] + 1)]
            if t[1]['fuel'] != 0:
                for place in options:
                    if ((x,y), place) in self._short_distances_paths.keys():
                        temp_texi += ((place),)
                for i in temp_texi:
                    taxis_places[index] += (('move', t[0], i),)
        # pickup
        pasenger_tup = state_dictionary["passengers"].items()
        for p in pasenger_tup:
            for index, t in enumerate(taxi_tup):
                if t[1]['location'] == p[1]['location'] and t[1]['capacity'] - 1 >= 0 and p[1]['location']!= p[1]['destination']:
                    taxis_places[index] += (('pick up', t[0], p[0]),)

        # dropoff
        for p in pasenger_tup:
            for index, t in enumerate(taxi_tup):
                if t[1]["location"] == p[1]['destination'] and t[0] == p[1]['location']:
                    taxis_places[index] += (('drop off', t[0], p[0]),)

        # wait+fuell
        for index, t in enumerate(taxi_tup):
            loc = t[1]["location"]
            x, y = loc[0],loc[1]
            if ((x,y) in self.gass_station):
                taxis_places[index] += (('refuel', t[0]),)
            taxis_places[index] += (('wait', t[0]),)

        for comb in itertools.product(*taxis_places):
            if taxi_num > 1:
                if self.dest_check(comb, taxi_c_loc, com):
                    actions_set += (comb,)
            else:
                actions_set += (comb,)
        list_actions_set = list(actions_set)
        list_actions_set.extend(['reset'])
        if state_dictionary['turns to go'] == 0:
            list_actions_set.extend(['terminate'])
        return list_actions_set

    def result(self, state, actions):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        if type(state) == dict:
            state_dictionary = state.copy()
        else:
            state_dictionary = json.loads(state)
        if actions == 'reset':
            initial_dictionary = json.loads(self.initial)
            initial_dictionary['turns to go'] = state_dictionary['turns to go'] - 1
            for p_name in state_dictionary['passengers'].keys():
                initial_dictionary['passengers'][p_name]['destination'] = state_dictionary['passengers'][p_name]['destination']
            return json.dumps(initial_dictionary)
        if actions == 'terminate':
            state_dictionary['turns to go'] = 0
            return json.dumps(state_dictionary)
        if len(actions) == 0:
            return state_dictionary
        if type(actions[0]) != tuple:
            actions = [actions]

        for atomic_action in actions:
            action_name = atomic_action[0]
            if action_name == 'move':
                state_dictionary['taxis'][atomic_action[1]]['location'] = atomic_action[2]
                state_dictionary['taxis'][atomic_action[1]]['fuel'] -= 1
            elif action_name == 'pick up':
                state_dictionary['taxis'][atomic_action[1]]['capacity'] -= 1
                state_dictionary['passengers'][atomic_action[2]]['location'] = atomic_action[1]
            elif action_name == 'drop off':
                state_dictionary['passengers'][atomic_action[2]]['location'] = state_dictionary['passengers'][atomic_action[2]]['destination']
                state_dictionary['taxis'][atomic_action[1]]['capacity'] += 1
            elif action_name == 'refuel':
                state_dictionary['taxis'][atomic_action[1]]['fuel'] = self.taxi_full_fuel[atomic_action[1]]
            elif action_name == 'wait':
                continue
        state_dictionary['turns to go'] -= 1
        return json.dumps(state_dictionary)

    def powerset(self, iterable):
        """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def legal_new_states(self, state):
        if type(state) == dict:
            state_dictrionary = copy.deepcopy(state)
        else:
            state_dictrionary = json.loads(state)

        state_legal_actions = self.actions(state_dictrionary)
        new_states=[]
        for act in state_legal_actions:
            if act != 'terminate' and act != 'reset':
                new_states.append((self.result(copy.deepcopy(state_dictrionary), act), act))
        return_states = []
        passengers_sets = list(self.powerset(state_dictrionary['passengers'].keys()))
        for new_state, act in new_states:
            for subset_p in passengers_sets:
                destinations_comb=[]
                for destination_change in subset_p:
                    for possible_goal in state_dictrionary['passengers'][destination_change]['possible_goals']:
                        if possible_goal != state_dictrionary['passengers'][destination_change]['destination']:
                            temp = [(possible_goal, destination_change)]
                            destinations_comb.append(temp)
                prob = self.get_prob_for_state(state_dictrionary,subset_p)
                possible_destination = list(itertools.product(*destinations_comb))
                for destination in possible_destination:
                    if type(new_state) == dict:
                        curr_state = new_state.copy()
                    else:
                        curr_state = json.loads(new_state)
                    for info in destination:
                        curr_state['passengers'][info[1]]['destination'] = info[0]
                    if (curr_state, act, prob) not in return_states:
                        return_states.append((json.dumps(curr_state), act, prob))
        initial_dict = json.loads(self.initial)
        initial_dict['turns to go'] = state_dictrionary['turns to go'] - 1
        return_states.append((json.dumps(initial_dict), 'reset', 1))
        return return_states

    def get_prob_for_state(self, state, subset_p):
        probs = []
        for p in state['passengers'].keys():
            if len(state['passengers'][p]['possible_goals']) == 1:
                probs.append(1)
            else:
                change_prob = state['passengers'][p]['prob_change_goal'] / len(
                    state['passengers'][p]['possible_goals'])
                if p in subset_p:
                    probs.append(change_prob)
                else:
                    probs.append(1 - state['passengers'][p]['prob_change_goal'] + change_prob)
        return np.prod(probs)

    def reward_action(self, actions):
        if actions == 'terminate':
            return 0
        if actions == 'reset':
            return -RESET_PENALTY
        score = 0
        for act in actions:
            if act[0] == 'refuel':
                score += -REFUEL_PENALTY
            elif act[0] == 'drop off':
                score += DROP_IN_DESTINATION_REWARD
        return score

    def MDP_init(self, start_state):
        q = queue.Queue()
        q.put(start_state)
        while not q.empty():
            temp_state = q.get()
            self.iteration_stack.put(temp_state)
            if temp_state in self.MDP.keys():
                continue
            state_dictionary = json.loads(temp_state)
            self.MDP[temp_state] = {}
            self.MDP[temp_state]['value'] = 0
            if state_dictionary['turns to go'] == 0:
                continue
            #(state,action,prob)
            new_states_info = self.legal_new_states(temp_state)
            for state, action, prob in new_states_info:
                reward = 0
                key_set = self.MDP[temp_state].keys()
                if action not in key_set:
                    self.MDP[temp_state][action] = {}
                state_w_r = {'weight': prob, 'reward': reward}
                self.MDP[temp_state][action][state] = state_w_r
                key_set2 = self.MDP.keys()
                if state not in key_set2:
                    q.put(state)

    def value_iteration(self, discount_factor):
        while not self.iteration_stack.empty():
            temp_max_value = 0
            state = self.iteration_stack.get()
            key_set=self.MDP.keys()
            if state not in key_set:
                continue
            for act in self.MDP[state].keys():
                if act == 'value' or act == 'reward':
                    continue
                sum_e_reward = self.reward_action(act)
                for next_state in self.MDP[state][act].keys():
                    if next_state == 'reward':
                        continue
                    sum_e_reward += self.MDP[state][act][next_state]['weight'] * self.MDP[next_state]['value']
                if sum_e_reward > temp_max_value:
                    temp_max_value = sum_e_reward
            self.MDP[state]['value'] = discount_factor * temp_max_value

    def our_policy(self):
        for state in self.MDP.keys():
            temp_max_value = 0
            best_action = 'terminate'
            for act in self.MDP[state].keys():
                if act == 'value' or act == 'reward':
                    continue
                sum_e_reward = self.reward_action(act)
                for new_state in self.MDP[state][act].keys():
                    if new_state == 'reward':
                        continue
                    sum_e_reward += self.MDP[state][act][new_state]['weight'] * self.MDP[new_state]['value']
                if sum_e_reward >= temp_max_value:
                    temp_max_value = sum_e_reward
                    best_action = act
            self.MDP[state]['best_action'] = best_action

    def act(self, state):
        state = json.dumps(state)
        return self.MDP[state]['best_action']


class TaxiAgent:
    def __init__(self, initial):
        self.initial = initial
        self.gass_station = ()
        self.G = self._build_graph(initial['map'])
        self._short_distances, self._short_distances_paths = self._create_shortest_path_distances(self.G)
        self.initial_turns_to_go = initial['turns to go']
        self.taxi_full_fuel = {taxi: initial['taxis'][taxi]['fuel'] for taxi in initial['taxis']}
        self.turns_to_deliver_first_passenger = []
        self.turns = 0
        self.dropped = 0
        self.lock = False
        self.flag = False
        self.cap_s = []

    def act(self, state):
        actions = self.taxis_action_builder(state)
        num_taxis = len((state["taxis"]))
        best_action = self.policy(state, actions, num_taxis)
        if best_action == 'reset':
            self.dropped = 0
            self.turns = 0
            self.lock = False
        for i in best_action:
            if i[0] == 'drop off':
                self.flag = True
        if best_action == 'terminate':
            return 'terminate'
        if(len(state['taxis'].keys())==1):
            return (best_action[0],)
        return best_action

    def dest_check(self, comb, taxi_c_loc, names):
        """
        Function that checks if the coordinates of the destination is legal
        """
        loc_set = []
        for i, ac in enumerate(comb):
            if ac[0] in names and taxi_c_loc[i] in loc_set:
                return False
            if ac[0] in names:
                loc_set.append(taxi_c_loc[i])
            if ac[0] == 'move' and ac[2] in loc_set:
                return False
            if ac[0] == 'move':
                loc_set.append(ac[2])
        return True

    def get_all_taxis_passengers(self, state):
        res = set()
        for t_passengers_list in state['taxi_passengers'].values():
            res.update(set(t_passengers_list))
        return res

    def find_closest_passenger(self, taxi_name, state):
        taxi_loc = state['taxis'][taxi_name]['location']
        taxi_fuel = state['taxis'][taxi_name]['fuel']
        m = 1000
        w = 'k'
        num_of_p = len(state["passengers"].keys())
        for p in state["passengers"].items():
            flag = False
            p_loc_destination = p[1]["location"]
            for t in state['taxis'].items():
                if p_loc_destination == t[0]:
                    flag = True
            if flag:
                continue
            key = taxi_loc, p_loc_destination
            dis_to_passenger = self._short_distances[key]
            temp_min = dis_to_passenger
            if temp_min == 0:
                if num_of_p > 1:
                    if p[1]["location"] == p[1]["destination"] and p[1]["prob_change_goal"] > 0.15:
                        return None
                    if p[1]["location"] == p[1]["destination"] and p[1]["prob_change_goal"] <= 0.15:
                        continue
                else:
                    return None
            if temp_min < m and temp_min < taxi_fuel:
                if num_of_p > 1:
                    if p[1]["location"] == p[1]["destination"] and p[1]["prob_change_goal"] <= 0.15:
                        continue
                m = temp_min
                w = p_loc_destination
        if m == 1000:
           for g in self.gass_station:
               key = taxi_loc, g
               temp_min = self._short_distances[key]
               if temp_min == 0:
                   return None
               if temp_min < m and temp_min <= taxi_fuel:
                   m = temp_min
                   w=g
        if w =='k':
            return None
        return w

    def find_best_destination(self, taxi_name, state):
        num_passengers_on_taxi = self.get_num_passengers_on_taxis1(state, taxi_name)
        if num_passengers_on_taxi == 0:
            return None
        taxi_location = state['taxis'][taxi_name]['location']
        taxi_fuel = state['taxis'][taxi_name]['fuel']
        m = 1000
        w = 'k'
        for p in state["passengers"].items():
            if p[1]['location'] == taxi_name:
                if p[1]['prob_change_goal'] < 0.1:
                    go_to = p[1]['destination']
                    if state['map'][go_to[0]][go_to[1]] == 'I':
                        continue
                    key = taxi_location, go_to
                    temp = self._short_distances[key]
                    if temp == 0:
                        return None
                    if m > temp and taxi_fuel >= temp:
                        return go_to
                    if m == 1000:
                        for g in self.gass_station:
                            key = taxi_location, g
                            temp_min = self._short_distances[key]
                            if temp_min == 0:
                                return None
                            if temp_min < m and temp_min <= taxi_fuel:
                                m = temp_min
                                w = g
                else:
                    n = 1000
                    for d in p[1]["possible_goals"]:
                        if state['map'][d[0]][d[1]] == 'I':
                            continue
                        key = taxi_location, d
                        temp = self._short_distances[key]
                        if temp == 0:
                            return None
                        if n > temp and taxi_fuel >= temp:
                            n=temp
                            w=d
                        if n == 1000:
                            for g in self.gass_station:
                                key = taxi_location, g
                                temp_min = self._short_distances[key]
                                if temp_min==0:
                                    return None
                                if temp_min < n and temp_min <= taxi_fuel:
                                    n = temp_min
                                    w=g
        if w == 'k':
            return None
        return w

    def get_num_passengers_on_taxis1(self, state, taxi_name):
        counter =0
        prob= None
        for p in state['passengers'].items():
            if p[1]['location'] == taxi_name:
                counter+=1
                prob=p[1]["prob_change_goal"]
                if p[1]["prob_change_goal"]>0.15:
                    return 1 ,None
        if counter == 0:
            return 0, None
        else:
            return 1, prob

    def find_best_move(self, taxi_name, state):
        num_passengers_on_taxi, prob = self.get_num_passengers_on_taxis1(state, taxi_name)
        taxi_location=state['taxis'][taxi_name]['location']
        if num_passengers_on_taxi == 0:
            taxi_destination = self.find_closest_passenger(taxi_name, state)
            if taxi_destination == None:
                return None
            if taxi_destination == taxi_location:
                return None
            edge = taxi_location, taxi_destination
            return ('move', taxi_name, self._short_distances_paths[edge][0])
        if num_passengers_on_taxi == 1:
            if prob != None and state['taxis'][taxi_name]["capacity"]>0:
                taxi_destination = self.find_closest_passenger(taxi_name, state)
                if taxi_destination == None:
                    return None
                if taxi_destination == taxi_location:
                    return None
                edge = taxi_location, taxi_destination
                return ('move', taxi_name, self._short_distances_paths[edge][0])
        taxi_passengers_destination = self.find_best_destination(taxi_name, state)
        if taxi_passengers_destination == None:
            return None
        edge = taxi_location, taxi_passengers_destination
        return ('move', taxi_name, self._short_distances_paths[edge][0])

    def num_passengers_on_taxis_policy(self, state):
        total = 0
        for t in state["taxis"].items():
            total = total+t[1]["capacity"]
        return sum(self.cap_s)-total

    def policy(self, state, actions, num_taxis):
        passengers_on_taxis = self.num_passengers_on_taxis_policy(state)
        passengers_on_map = len(state['passengers']) - passengers_on_taxis
        if self.flag:
            return 'terminate'
        if passengers_on_map == 0 and passengers_on_taxis == 0:
            if self.dropped >0:
                return 'terminate'
            else:
                average_turns_to_2_deliver = sum(self.turns_to_deliver_first_passenger) / len(
                    self.turns_to_deliver_first_passenger)
                if state['turns to go'] >= average_turns_to_2_deliver:
                    self.turns += 1
                    return 'reset'
                else:
                    return 'terminate'

        best_action, best_score = None, None
        for action in actions:
            score = 0
            if num_taxis == 1:
                if action[0] == 'move':
                    score += 1
                if action[0] == 'drop off':
                    score += 1000
                if action[0] == 'pick up':
                    score += 100
                if action[0] =='refuel':
                    if state['taxis'][action[1]]['fuel']< self.initial['taxis'][action[1]]['fuel']:
                        score += 1000
                score = score - passengers_on_taxis * PASSENGERS_ON_TAXI_WEIGHT - \
                        passengers_on_map * PASSENGERS_ON_MAP_WEIGHT
                if best_score is None or score > best_score:
                    best_score, best_action = score, action
            else:
                total_score = 0
                for act in action:
                    if act[0] == 'move':
                        score += 1
                    if act[0] == 'drop off':
                        score += 1000
                    if act[0] == 'pick up':
                        score += 100
                    if act[0] =='refuel':
                        if state['taxis'][act[1]]['fuel'] < self.initial['taxis'][act[1]]['fuel']-1:
                            score += 90
                    score = score - passengers_on_taxis * PASSENGERS_ON_TAXI_WEIGHT - \
                            passengers_on_map * PASSENGERS_ON_MAP_WEIGHT
                    total_score += score
                if best_score is None or total_score > best_score:
                    best_score, best_action = total_score, action
        self.turns += 1
        if num_taxis == 1:
            if best_action[0] == 'drop off':
                self.dropped += 1
            return [best_action]
        for act in best_action:
            if act[0] == 'drop off':
                self.dropped += 1
        return best_action

    def taxis_action_builder(self, state):
        """
        Builds all possible actions from a given state
        :param state: game state
        :return: a tuple of possible actions where each action is represented by a tuple
        """
        action_names = ('pick up', 'drop off', 'wait', 'refuel')
        taxis_places = []
        taxi_c_loc = []
        taxi_tup = state["taxis"].items()
        actions_set = ()
        for i, t in enumerate(taxi_tup):
            taxis_places.append(())
            taxi_c_loc.append(t[1]["location"])
        for index, t in enumerate(taxi_tup):
            if t[1]['fuel'] != 0:
                m=self.find_best_move(t[0], state )
                if m is not None:
                    taxis_places[index] += (m,)
        # pickup
        pasenger_tup = state["passengers"].items()
        for p in pasenger_tup:
            for index, t in enumerate(taxi_tup):
                if t[1]['location'] == p[1]["location"] and t[1]["capacity"] - 1 >= 0 and p[1]["destination"] != p[1]["location"]:
                    taxis_places[index] += (('pick up', t[0], p[0]),)
        # dropoff
        for p in pasenger_tup:
            for index, t in enumerate(taxi_tup):
                if t[1]["location"] == p[1]["destination"] and t[0] == p[1]["location"]:
                    taxis_places[index] += (('drop off', t[0], p[0]),)
                    self.dropped += 1
        # wait+refuel
        for index, t in enumerate(taxi_tup):
            if (t[1]["location"] in self.gass_station) and (t[1]['fuel'] != self.taxi_full_fuel[t[0]]):
                taxis_places[index] += (('refuel', t[0]),)
            taxis_places[index] += (('wait', t[0]),)

        for comb in itertools.product(*taxis_places):
            if self.dest_check(comb, taxi_c_loc, action_names):
                actions_set += (comb,)

        if(len(state['taxis'].keys())==1):
            return taxis_places[0]
        return actions_set

    def _build_graph(self, game_map):
        """
        Builds the game graph where a node is represented as (x,y) coordinate, and 'I' is not reachable node
        :param game_map: the map of the game
        :return: graph representing the game map
        """
        G = nx.Graph()
        rows, cols = len(game_map), len(game_map[0])
        for i in range(rows):
            for j in range(cols):
                if game_map[i][j] == 'I':
                    continue
                if game_map[i][j] == 'G':
                    self.gass_station += ((i, j),)
                # edge from (i,j) to its adjacent
                if i + 1 < rows and game_map[i + 1][j] != 'I':
                    G.add_edge((i, j), (i + 1, j))
                if i - 1 >= 0 and game_map[i - 1][j] != 'I':
                    G.add_edge((i, j), (i - 1, j))
                if j + 1 < cols and game_map[i][j + 1] != 'I':
                    G.add_edge((i, j), (i, j + 1))
                if j - 1 >= 0 and game_map[i][j - 1] != 'I':
                    G.add_edge((i, j), (i, j - 1))
        return G

    def _create_shortest_path_distances(self, G):
        """
        Creates shortest paths dictrionary
        :param G: graph object
        :return: dictrionary of shortest paths
        """
        paths_len, paths = {}, {}
        for n1 in G.nodes:
            for n2 in G.nodes:
                if n1 != n2:
                    path = nx.shortest_path(G, n1, n2)[1:]
                    paths[(n1, n2)] = path
                    paths_len[(n1, n2)] = len(path)
                else:
                    paths[(n1, n2)] = []
                    paths_len[(n1, n2)] = 0
        return paths_len, paths

