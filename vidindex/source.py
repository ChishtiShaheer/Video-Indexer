"""Reading videos out of FineVideo parquet files (one row = one mp4)."""
import glob
import os

from .common import short_error


def find_parquet_files(path):
    """A single .parquet file, or every *.parquet below a folder (sorted by name)."""
    if os.path.isfile(path):
        return [path]
    files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
    return sorted(files, key=lambda p: (os.path.basename(p), p))


def iter_videos(files, skip=frozenset(), limit=None):
    """Yield (video_id, mp4_bytes_or_None, error_or_None) for every row.

    video_id is "<parquet file name>:<row number counted from 0>".
    Rows that cannot be read are yielded with data=None and an error message so
    the caller can write an error line.  Video ids in `skip` are not yielded.
    """
    import pyarrow.parquet as pq

    yielded = 0
    for path in files:
        name = os.path.basename(path)
        try:
            pf = pq.ParquetFile(path)
            total = pf.metadata.num_rows
            if "mp4" not in pf.schema_arrow.names:
                raise ValueError("no 'mp4' column")
        except Exception as e:
            yield "%s:0" % name, None, "could not read parquet file (%s)" % short_error(e, 100)
            continue

        row = 0
        try:
            for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
                vid = "%s:%d" % (name, row)
                row += 1
                if vid in skip:
                    continue
                if limit is not None and yielded >= limit:
                    return
                yielded += 1
                try:
                    data = batch.column(0)[0].as_py()
                except Exception as e:
                    yield vid, None, "could not read row (%s)" % short_error(e, 100)
                    continue
                yield vid, data, None
        except Exception as e:
            # the rest of this file is unreadable: one error line per remaining row
            msg = "parquet read error (%s)" % short_error(e, 100)
            for r in range(row, max(total, row + 1)):
                vid = "%s:%d" % (name, r)
                if vid not in skip:
                    yield vid, None, msg
