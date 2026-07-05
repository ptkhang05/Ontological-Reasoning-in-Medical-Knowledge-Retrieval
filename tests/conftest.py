from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from clinical_nlp.api.app import create_app
from clinical_nlp.pipeline import ClinicalPipeline
from clinical_nlp.terminology import TerminologyStore


@pytest.fixture()
def client() -> Iterator[TestClient]:
    pipeline = ClinicalPipeline(terminology=TerminologyStore.default())
    with TestClient(create_app(pipeline)) as test_client:
        yield test_client
