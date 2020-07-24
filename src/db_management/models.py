import datetime

import mongoengine
from mongoengine.document import Document, EmbeddedDocument
from mongoengine.fields import (EmbeddedDocumentListField, EmbeddedDocumentField,
                                FloatField, DateTimeField, IntField, StringField,
                                ReferenceField, ListField, MapField)


class MetricHistoryRecord(EmbeddedDocument):
    epoch = IntField(required=True)
    value = FloatField(required=True)

    created_at = DateTimeField(required=True, default=datetime.datetime.now)
    updated_at = DateTimeField(required=True, default=datetime.datetime.now)


class MetricResult(EmbeddedDocument):
    value = FloatField(required=True)
    std = FloatField()


class TrainedModel(Document):
    machine_name = StringField(required=True)
    model_name = StringField(required=True)
    dataset_name = StringField(required=True)
    run = IntField(required=True)
    train_temperature = StringField(required=True)

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
                    'train_temperature'
                ],
                'unique': True,
            },
            '#machine_name',
            '#model_name',
            '#dataset_name',
            '+run',
            '#train_temperature',
            '-created_at',
            '-updated_at',
        ],
    }

    def clean(self):
        self.model_name = self.model_name.lower()
        self.dataset_name = self.dataset_name.lower()
        self.train_temperature = self.train_temperature.lower()
        self.updated_at = datetime.datetime.now()
        super().clean()


class EvaluatedModel(Document):
    trained_model = ReferenceField(TrainedModel, required=True,
                                   reverse_delete_rule=mongoengine.CASCADE)

    restore_type = StringField(required=True)
    test_temperature = StringField(required=True)

    metrics = MapField(EmbeddedDocumentField(MetricResult), required=True)

    created_at = DateTimeField(required=True, default=datetime.datetime.now)
    updated_at = DateTimeField(required=True, default=datetime.datetime.now)

    meta = {
        'indexes': [
            {
                'fields': [
                    'trained_model',
                    'restore_type',
                    '+test_temperature',
                ],
                'unique': True
            },
            'trained_model',
            '#restore_type',
            '+test_temperature',
            '-created_at',
            '-updated_at',
        ],
    }

    def clean(self):
        self.restore_type = self.restore_type.lower()
        self.test_temperature = self.test_temperature.lower()
        self.updated_at = datetime.datetime.now()
        super().clean()


class Sample(Document):
    evaluated_model = ReferenceField(EvaluatedModel, required=True,
                                     reverse_delete_rule=mongoengine.CASCADE)

    index = IntField(required=True)
    origin = StringField()

    tokens = ListField(StringField(), required=True)
    sentence = StringField(required=True)
    metrics = MapField(EmbeddedDocumentField(MetricResult))

    meta = {
        'indexes': [
            {
                'fields': [
                    'evaluated_model',
                    '+index',
                    'origin'
                ],
                'unique': True
            },
            'evaluated_model',
            '+index',
            '#origin',
        ]
    }

    def clean(self):
        self.origin = self.origin.lower()
        assert self.origin in ('test', 'generated')


import db_management.setup


print('TrainedModel documents: {}'.format(TrainedModel.objects().count()))
print('EvaluatedModel documents: {}'.format(EvaluatedModel.objects().count()))
print('Sample documents: {}'.format(Sample.objects().count()))
