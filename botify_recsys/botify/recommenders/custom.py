from .random import Random
from .recommender import Recommender
from .indexed import Indexed
import random
# import torch
# import faiss
from collections import defaultdict
# from transformers import AutoTokenizer, AutoModel
# import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# model = AutoModel.from_pretrained(model_ckpt)
# device = torch.device("cpu")
# model.to(device)


class Custom(Recommender):
    """
    Recommend tracks closest to the previous one.
    Fall back to the random recommender if no
    recommendations found for the track.
    """

    def __init__(self, tracks_redis, recommendations_redis, catalog, ranked, used):
        self.tracks_redis = tracks_redis
        self.random = Random(tracks_redis)
        self.fallback = Indexed(tracks_redis, recommendations_redis, catalog)
        self.catalog = catalog
        self.ranked = ranked
        self.used = used

    @staticmethod
    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(self, tracks_list):
        encoded_input = tokenizer(
            [str(val) for val in tracks_list], padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        return self.cls_pooling(model_output)


    # TODO Seminar 5 step 1: Implement contextual recommender based on NN predictions
    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if user not in self.used:
            self.used[user] = []
        self.used[user].append(prev_track)
        if user not in self.ranked:
            self.ranked[user] = {}
        self.ranked[user] = defaultdict(int)
        if prev_track_time > 0.5:
            self.ranked[user][prev_track] += 1
        elif len(self.ranked[user]) > 0:
            prev_track, pos = random.choice(list(self.ranked[user].items()))
            self.ranked[user][prev_track] = pos + 1
        else:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        previous_track = self.tracks_redis.get(prev_track)

        if previous_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        previous_track = self.catalog.from_bytes(previous_track)
        recommendations = previous_track.recommendations
        if recommendations is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        shuffled = list(recommendations)
        random.shuffle(shuffled)
        return shuffled[0]

