import datetime

import mongoengine
from mongoengine.document import Document, EmbeddedDocument
from mongoengine.fields import (EmbeddedDocumentListField, EmbeddedDocumentField,
                                FloatField, DateTimeField, IntField, StringField,
                                ReferenceField, ListField, MapField)

# import db_management.setup


class MetricHistoryRecord(EmbeddedDocument):
    epoch = IntField(required=True)
    value = FloatField(required=True)

    created_at = DateTimeField(required=True, default=datetime.datetime.now)
    updated_at = DateTimeField(required=True, default=datetime.datetime.now)


class InTrainingEvaluationHistory(Document):
    machine_name = StringField(required=True)
    model_name = StringField(required=True)
    dataset_name = StringField(required=True)
    run = IntField(required=True)

    all_history = MapField(EmbeddedDocumentListField(MetricHistoryRecord))
    best_history = MapField(EmbeddedDocumentListField(MetricHistoryRecord))

    created_at = DateTimeField(required=True, default=datetime.datetime.now)
    updated_at = DateTimeField(required=True, default=datetime.datetime.now)

    meta = meta = {
        'indexes': [
            {
                'fields': [
                    'model_name',
                    'dataset_name',
                    '+run',
                ],
                'unique': True,
            },
            '#machine_name',
            '#model_name',
            '#dataset_name',
            '+run',
            '-created_at',
            '-updated_at',
        ],
    }

    def clean(self):
        self.model_name = self.model_name.lower()
        self.dataset_name = self.dataset_name.lower()


class Model(Document):
    machine_name = StringField(required=True)
    model_name = StringField(required=True)
    dataset_name = StringField(required=True)
    run = IntField(required=True)
    restore_type = StringField(required=True)
    temperature = StringField(required=True)

    created_at = DateTimeField(required=True, default=datetime.datetime.now)
    updated_at = DateTimeField(required=True, default=datetime.datetime.now)

    meta = {
        'indexes': [
            '#machine_name',
            '#model_name',
            '#dataset_name',
            '#restore_type',
            '+run',
            '+temperature',
            '-created_at',
            '-updated_at',
            {
                'fields': [
                    'model_name',
                    'dataset_name',
                    'restore_type',
                    '+run',
                    '+temperature',
                ],
                'unique': True
            }
        ],
    }

    def clean(self):
        self.model_name = self.model_name.lower()
        self.dataset_name = self.dataset_name.lower()
        self.restore_type = self.restore_type.lower()
        self.temperature = self.temperature.lower()


class MetricResult(EmbeddedDocument):
    value = FloatField(required=True)
    std = FloatField()


class Sample(Document):
    model = ReferenceField(Model, required=True, reverse_delete_rule=mongoengine.CASCADE)

    index = IntField(required=True)
    origin = StringField()

    tokens = ListField(StringField(), required=True)
    sentence = StringField(required=True)
    metrics = MapField(EmbeddedDocumentField(MetricResult))

    meta = {
        'indexes': [
            'model',
            '+index',
            '#origin',
            {
                'fields': [
                    'model',
                    '+index',
                    'origin'
                ],
                'unique': True
            }
        ]
    }

    def clean(self):
        self.origin = self.origin.lower()
        assert self.origin in ('test', 'generated')


class ModelEvaluationResult(Document):
    model = ReferenceField(Model, required=True,
                           reverse_delete_rule=mongoengine.CASCADE, unique=True)
    metrics = MapField(EmbeddedDocumentField(MetricResult), required=True)

    created_at = DateTimeField(required=True, default=datetime.datetime.now)
    updated_at = DateTimeField(required=True, default=datetime.datetime.now)

    meta = {
        'indexes': [
            'model',
            '-created_at',
            '-updated_at'
        ]
    }


print('InTrainingEvaluationHistory documents: {}'.format(
    InTrainingEvaluationHistory.objects().count()))
print('Model documents: {}'.format(Model.objects().count()))
print('Sample documents: {}'.format(Sample.objects().count()))
print('ModelEvaluationResult documents: {}'.format(ModelEvaluationResult.objects().count()))
