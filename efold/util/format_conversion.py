# this code wsa taken from arnie_utils.py


def convert_bp_list_to_dotbracket(bp_list, seq_len):
    
    # convert the bp list from 1-indexed to 0-indexed
    bp_list = [(b-1, c-1) for b, c in bp_list]
    
    db = "." * seq_len
    # group into bps that are not intertwined and can use same brackets!
    groups = _group_into_non_conflicting_bp(bp_list)

    # all bp that are not intertwined get (), but all others are
    # groups to be nonconflicting and then asigned (), [], {}, <> by group
    chars_set = [("(", ")"), ("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
    alphabet = [(chr(lower), chr(upper))
                for upper, lower in zip(list(range(65, 91)), list(range(97, 123)))]
    chars_set.extend(alphabet)

    if len(groups) > len(chars_set):
        print("WARNING: PK too complex, not enough brackets to represent it.")
        return None

    for group, chars in zip(groups, chars_set):
        for bp in group:
            db = db[:bp[0]] + chars[0] + \
                db[bp[0] + 1:bp[1]] + chars[1] + db[bp[1] + 1:]
    return db


def _group_into_non_conflicting_bp(bp_list):
    ''' given a bp_list, group basepairs into groups that do not conflict

    Args
            bp_list: list of base_pairs

    Returns:
            groups of baspairs that are not intertwined
    '''
    conflict_list = _get_list_bp_conflicts(bp_list)

    non_redudant_bp_list = _get_non_redudant_bp_list(conflict_list)
    bp_with_no_conflict = [
        bp for bp in bp_list if bp not in non_redudant_bp_list]
    groups = [bp_with_no_conflict]
    while non_redudant_bp_list != []:
        current_bp = non_redudant_bp_list[0]
        current_bp_conflicts = []
        for conflict in conflict_list:
            if current_bp == conflict[0]:
                current_bp_conflicts.append(conflict[1])
            elif current_bp == conflict[1]:
                current_bp_conflicts.append(conflict[0])
        max_group = [
            bp for bp in non_redudant_bp_list if bp not in current_bp_conflicts]
        to_remove = []
        for i, bpA in enumerate(max_group):
            for bpB in max_group[i:]:
                if bpA not in to_remove and bpB not in to_remove:
                    if [bpA, bpB] in conflict_list or [bpB, bpA] in conflict_list:
                        to_remove.append(bpB)
        group = [bp for bp in max_group if bp not in to_remove]
        groups.append(group)
        non_redudant_bp_list = current_bp_conflicts
        conflict_list = [conflict for conflict in conflict_list if conflict[0]
                         not in group and conflict[1] not in group]
    return groups


def _get_non_redudant_bp_list(conflict_list):
    ''' given a conflict list get the list of nonredundant basepairs this list has

    Args:
            conflict_list: list of pairs of base_pairs that are intertwined basepairs
    returns:
            list of basepairs in conflict list without repeats
    '''
    non_redudant_bp_list = []
    for conflict in conflict_list:
        if conflict[0] not in non_redudant_bp_list:
            non_redudant_bp_list.append(conflict[0])
        if conflict[1] not in non_redudant_bp_list:
            non_redudant_bp_list.append(conflict[1])
    return non_redudant_bp_list



def _get_list_bp_conflicts(bp_list):
    '''given a bp_list gives the list of conflicts bp-s which indicate PK structure
    Args:
            bp_list: of list of base pairs where the base pairs are list of indeces of the bp in increasing order (bp[0]<bp[1])
    returns:
            List of conflicting basepairs, where conflicting is pairs of base pairs that are intertwined.
    '''
    if len(bp_list) <= 1:
        return []
    else:
        current_bp = bp_list[0]
        conflicts = []
        for bp in bp_list[1:]:
            if (bp[0] < current_bp[1] and current_bp[1] < bp[1]):
                conflicts.append([current_bp, bp])
        return conflicts + _get_list_bp_conflicts(bp_list[1:])
