from tensorboard.summary.writer import event_file_writer

from skrl.utils.tensorboard import SummaryWriter


def test_writer_class_is_resolved_at_instantiation(tmp_path, monkeypatch):
    """Integrations like W&B ``sync_tensorboard`` replace the module's ``EventFileWriter`` after skrl is imported."""
    created = []
    original = event_file_writer.EventFileWriter

    class PatchedEventFileWriter(original):
        def __init__(self, logdir, *args, **kwargs):
            created.append(logdir)
            super().__init__(logdir, *args, **kwargs)

    monkeypatch.setattr(event_file_writer, "EventFileWriter", PatchedEventFileWriter)
    writer = SummaryWriter(log_dir=str(tmp_path))
    writer.add_scalar(tag="a", value=1.0, timestep=0)
    writer.close()
    assert created == [str(tmp_path)]
    assert any(path.name.startswith("events.out.tfevents") for path in tmp_path.iterdir())
