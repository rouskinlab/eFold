class Stack:
    def __init__(self, L, mode, data_type):
        self.L = L
        self.vals = []
        assert mode in ["best", "worse"], "mode must be in ['best','worse']"
        self.mode = mode
        self.data_type = data_type

    def _sort(self):
        self.vals.sort(
            key=lambda x: x["score_{}".format(self.data_type)],
            reverse=self.mode == "best",
        )

    def _should_replace(self, current_score, new_score):
        if self.mode == "best":
            return current_score < new_score
        return current_score > new_score

    def try_to_add(self, line):
        if len(self.vals) < self.L:
            self.vals.append(line)
            self._sort()
            return True

        elif self._should_replace(
            self.vals[-1]["score_{}".format(self.data_type)],
            line["score_{}".format(self.data_type)],
        ):
            self.vals.pop()
            self.vals.append(line)
            self._sort()
            return True
        return False

    def is_empty(self):
        return not len(self.vals)
