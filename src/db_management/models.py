from mongoengine.document import Document, EmbeddedDocument
from mongoengine.fields import (EmbeddedDocumentListField, FloatField,
                                ListField, MapField, StringField)

import db_management.setup


class Model(Document):
    machine_name = StringField(required=True)
    model_name = StringField(required=True)
    dataset_name = StringField(required=True)
    run = StringField()
    restore_type = StringField()
    temperature = FloatField()
    meta = {
        'abstract': True,
        'indexes': [
            '$machine_name'
            '$model_name',
            '$dataset_name',
            '$restore_type'
            '+run',
            '+temperature',
        ],
    }


class MetricHistoryRecord(EmbeddedDocument):
    epoch = StringField(required=True)
    value = FloatField(required=True)


class InTrainingEvaluationHistory(Model):
    all_history = MapField(EmbeddedDocumentListField(MetricHistoryRecord))
    best_history = MapField(EmbeddedDocumentListField(MetricHistoryRecord))
    meta = {
        'indexes': Model.meta['indexes']
    }


class MetricResult(EmbeddedDocument):
    value = FloatField(required=True)
    std = FloatField()


class Sample(EmbeddedDocument):
    tokens = ListField(StringField(), required=True)
    sentence = StringField(required=True)
    metrics = MapField(MetricResult, required=True)
    # other supplementary info can be placed here (dynamic fields)


class ModelSamples(Model):
    generated_samples = EmbeddedDocumentListField(Sample, required=True)
    test_samples = EmbeddedDocumentListField(Sample, required=True)
    meta = {
        'indexes': Model.meta['indexes']
    }


class ModelEvaluationResult(Model):
    metrics = MapField(MetricResult, required=True)
    meta = {
        'indexes': Model.meta['indexes']
    }

# print(ModelEvaluationResult.objects)
