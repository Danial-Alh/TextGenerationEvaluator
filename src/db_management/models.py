from mongoengine.document import Document, EmbeddedDocument
from mongoengine.fields import (EmbeddedDocumentListField, EmbeddedDocumentField,
                                FloatField, DateTimeField, IntField, StringField,
                                ListField, MapField)
import datetime
# import db_management.setup


class Model(Document):
    machine_name = StringField(required=True)
    model_name = StringField(required=True)
    dataset_name = StringField(required=True)
    run = IntField()
    restore_type = StringField()
    temperature = StringField()

    created_at = DateTimeField(required=True, default=datetime.datetime.now)
    updated_at = DateTimeField(required=True, default=datetime.datetime.now)

    meta = {
        'abstract': True,
        # 'allow_inheritence': True
    }
    TEMP_META = {
        'indexes': [
            '#machine_name',
            '#model_name',
            '#dataset_name',
            '#restore_type',
            '+run',
            '+temperature',
            '-created_at',
            '-updated_at',
        ],
    }


class MetricHistoryRecord(EmbeddedDocument):
    epoch = IntField(required=True)
    value = FloatField(required=True)

    created_at = DateTimeField(required=True, default=datetime.datetime.now)
    updated_at = DateTimeField(required=True, default=datetime.datetime.now)


class InTrainingEvaluationHistory(Model):
    all_history = MapField(EmbeddedDocumentListField(MetricHistoryRecord))
    best_history = MapField(EmbeddedDocumentListField(MetricHistoryRecord))

    meta = {
        'indexes': Model.TEMP_META['indexes']
    }


class MetricResult(EmbeddedDocument):
    value = FloatField(required=True)
    std = FloatField()


class Sample(EmbeddedDocument):
    tokens = ListField(StringField(), required=True)
    sentence = StringField(required=True)
    metrics = MapField(EmbeddedDocumentField(MetricResult))
    # other supplementary info can be placed here (dynamic fields)


class ModelSamples(Model):
    generated_samples = EmbeddedDocumentListField(Sample)
    test_samples = EmbeddedDocumentListField(Sample)

    meta = {
        'indexes': Model.TEMP_META['indexes']
    }


class ModelEvaluationResult(Model):
    metrics = MapField(EmbeddedDocumentField(MetricResult), required=True)

    meta = {
        'indexes': Model.TEMP_META['indexes']
    }


print('InTrainingEvaluationHistory documents: {}'.format(InTrainingEvaluationHistory.objects().count()))
print('ModelSamples documents: {}'.format(ModelSamples.objects().count()))
print('ModelEvaluationResult documents: {}'.format(ModelEvaluationResult.objects().count()))
