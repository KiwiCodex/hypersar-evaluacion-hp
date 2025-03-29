import numpy as np
import torch
from datetime import datetime

default_value = 0.0


def precision(correct_predictions, k):
    num_hit = torch.sum(correct_predictions, dim=-1)
    return num_hit / k

def recall(correct_predictions, num_relevant):
    num_hit = torch.sum(correct_predictions, dim=-1)
    return num_hit / num_relevant

def ndcg(correct_predictions, num_relevant, k):
    ideal_correct_predictions = torch.zeros_like(correct_predictions)
    batch_size = ideal_correct_predictions.shape[0]
    for sample in range(batch_size):
        ideal_correct_predictions[sample, :num_relevant[sample]] = 1
    return dcg(correct_predictions, k) / dcg(ideal_correct_predictions, k)

def dcg(correct_predictions, k):
    result = 0.0
    for rank in range(k):
        result += correct_predictions[:, rank] / np.log2(rank + 2)
    return result

def map(correct_predictions, num_relevant, k):
    result = 0.0
    for rank in range(k):
        result += precision(correct_predictions[:, :rank + 1], rank + 1) * correct_predictions[:, rank]
    result /= num_relevant
    return result

def mae(correct_predicted_interactions, num_true_interactions, k):
    num_true_interactions = num_true_interactions.float() + 1e-8
    absolute_errors = torch.abs(correct_predicted_interactions[:, :k] - num_true_interactions.unsqueeze(-1))
    
    mae_values = torch.sum(absolute_errors, dim=-1) / num_true_interactions
    
    scaled_mae_values = mae_values / torch.max(mae_values)
    
    return scaled_mae_values

# def novelty(item_ids, known_interactions, k, scaling_factor=1.0): #scaling_factor modifica el grado de sensibilidad del Novelty
#     num_items = len(item_ids)
#     item_novelty = torch.zeros(num_items, dtype=torch.float, device=item_ids.device)

#     for i, item_id in enumerate(item_ids):
#         item_id = item_id.item() if torch.is_tensor(item_id) else item_id  

#         if item_id in known_interactions:
#             interactions = known_interactions[item_id]
            
#             if isinstance(interactions, list):
#                 interactions_dict = dict(interactions)
#             elif isinstance(interactions, dict):
#                 interactions_dict = interactions
#             else:
#                 print(f"Tipo de interacción no soportado: {type(interactions)}")
#                 continue

#             for user, count in interactions_dict.items():
#                 item_novelty[i] += scaling_factor / np.log2(count + 2)

#     # Realiza escala 0 a 1
#     min_value = torch.min(item_novelty)
#     max_value = torch.max(item_novelty)
#     item_novelty_scaled = (item_novelty - min_value) / (max_value - min_value)

#     return item_novelty_scaled


def hit_rate(correct_predictions, num_true_interactions, k):
    hits = torch.sum(correct_predictions, dim=1)
    hit_rate_values = hits.float() / num_true_interactions.float()
    hit_rate_values[torch.isnan(hit_rate_values)] = 0.0  

    return hit_rate_values


def evaluate(correct_predicted_interactions, num_true_interactions, metrics, item_ids, known_interactions):
    """
    Evaluates a ranking model in terms of precision and recall for the given cutoff values
    Args:
        correct_predicted_interactions: (array<bool>: n_rows * max(cutoffs)) 1 iff prediction matches a true interaction
        num_true_interactions: (array<bool>: n_rows) number of true interactions associated to each row
        metrics: (list<tuple<string,int>>) list of metrics to consider, with tuples made of the metric type and cutoff
        item_ids: (tensor) tensor of item IDs
        known_interactions: (dict) dictionary of known interactions

    Returns:
        eval_results: dictionary with evaluation results for each metric cumulated over all rows; keys are the metrics
    """
    eval_results = {}
    for metric in metrics:
        (metric_type, k) = metric
        correct_predictions = correct_predicted_interactions[:, :k]
        k = min(k, correct_predictions.shape[1])
        if metric_type == "precision":
            eval_results[metric] = precision(correct_predictions, k)
        elif metric_type == "recall":
            eval_results[metric] = recall(correct_predictions, num_true_interactions)
        elif metric_type == "ndcg":
            eval_results[metric] = ndcg(correct_predictions, num_true_interactions, k)
        elif metric_type == "dcg":
            eval_results[metric] = dcg(correct_predictions, k)
        elif metric_type == "map":
            eval_results[metric] = map(correct_predictions, num_true_interactions, k)
        elif metric_type == "mae":
            eval_results[metric] = mae(correct_predicted_interactions, num_true_interactions, k)
        # elif metric_type == "novelty":
        #     eval_results[metric] = novelty(item_ids, known_interactions, k)
        elif metric_type == "hit_rate":
            eval_results[metric] = hit_rate(correct_predictions, num_true_interactions, k)
    
    return eval_results




def predict_evaluate(data_loader, options, model, known_interactions):
    max_k = max([metric[1] for metric in options.metrics])
    max_k = min(max_k, options.num_item)
    types = ['all', 'rec', 'search']
    eval_results = {type: {metric: torch.tensor([], dtype=torch.float, device=options.device_ops)
                           for metric in options.metrics + [("mae", 1), ("hit_rate", 1)]} for type in types}

    for batch_id, batch in enumerate(data_loader):
        if batch_id % 10 == 0:
            print("Number of batches processed: {}/{}...".format(batch_id, len(data_loader)), datetime.now(), flush=True)

        device_embed = options.device_embed
        device_ops = options.device_ops
        user_ids = batch['user_ids'].to(device_embed)
        item_ids = batch['item_ids'].to(device_ops)
        interaction_types = batch['interaction_types'].to(device_ops)
        batch_size = len(user_ids)

        # Predict the items interacted for each user and mask the items which appeared in known interactions
        if options.model in ["FactorizationMachine", "DeepFM", "JSR", "DREM", "HyperSaR"]:
            keyword_ids = batch['keyword_ids'].to(device_embed)
            query_sizes = batch['query_sizes'].to(device_ops)
            predicted_scores = model.predict(user_ids, keyword_ids, query_sizes)
        else:
            predicted_scores = model.predict(user_ids)

        # Mask for each user the items from their training set
        mask_value = -np.inf
        for i, user in enumerate(user_ids):
            if int(user) in known_interactions:
                for interaction in known_interactions[int(user)]:
                    item = interaction[0]
                    predicted_scores[i, item] = mask_value

        _, predicted_interactions = torch.topk(predicted_scores, k=max_k, dim=1, largest=True, sorted=True)

        # Identify the correct interactions in the top-k predicted items
        correct_predicted_interactions = (predicted_interactions == item_ids.unsqueeze(-1)).float()

        # 1 relevant item
        num_true_interactions = torch.ones([batch_size], dtype=torch.long, device=options.device_ops)

        # Perform the evaluation
        batch_results = {}
        batch_results['all'] = evaluate(correct_predicted_interactions, num_true_interactions,
                                        options.metrics + [("mae", 1), ("hit_rate", 1)], item_ids, known_interactions)

        # Aggregating metrics for rec type
        recommendation_ids = torch.where(interaction_types == 0)[0]
        batch_results['rec'] = {
            metric: batch_results['all'][metric][recommendation_ids] if len(batch_results['all'][metric].shape) > 0
            else torch.tensor([default_value], dtype=torch.float, device=options.device_ops)
            for metric in options.metrics + [("mae", 1),("hit_rate", 1)]
        }

        # Aggregating metrics for search type
        search_ids = torch.where(interaction_types == 1)[0]
        batch_results['search'] = {
            metric: batch_results['all'][metric][search_ids] if len(batch_results['all'][metric].shape) > 0
            else torch.tensor([default_value], dtype=torch.float, device=options.device_ops)
            for metric in options.metrics + [("mae", 1), ("hit_rate", 1)]
        }

        eval_results = {
            type: {
                metric: torch.cat((eval_results[type][metric], batch_results[type][metric]), dim=0)
                if len(batch_results[type][metric].shape) > 0 and not torch.isnan(batch_results[type][metric]).any()
                else torch.tensor([], dtype=torch.float, device=options.device_ops)
                for metric in options.metrics + [("mae", 1), ("hit_rate", 1)]
            } for type in types
        }

    eval_results = {
        type: {
            metric: torch.mean(eval_results[type][metric], dim=0) for metric in options.metrics + [("mae", 1), ("hit_rate", 1)]
        } for type in types
    }

    return eval_results
