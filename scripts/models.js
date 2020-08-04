db.getCollection('trained_model')
.aggregate([
{
    $lookup:
    {
        from: "evaluated_model",
        localField: "_id",
        foreignField: "trained_model",
        as: "evaluated_models"
    }
},
{$unwind: "$evaluated_models"},
{$sort: {"dataset_name": +1, "evaluated_models.test_temperature": -1}},
])