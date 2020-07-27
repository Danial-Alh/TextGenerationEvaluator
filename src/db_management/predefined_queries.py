LEFT_JOIN_QUERY = [
    {
        "$lookup":
            {
                "from": "evaluated_model",
                "localField": "_id",
                "foreignField": "trained_model",
                "as": "evaluated_models"
            }
    },
    {
        "$unwind": "$evaluated_models"
    },
    {
        "$addFields":
            {
                "evaluated": {"$gt": [{"$size": {"$objectToArray": "$evaluated_models.metrics"}}, 0]},
                "test_temperature": "$evaluated_models.test_temperature",
                "restore_type": "$evaluated_models.restore_type"
            }
    },
    {
        "$project":
            {
                "_id": False, "all_history": False, "best_history": False,
                "created_at": False, "updated_at": False,
                "evaluated_models": False
            }
    },

]
