import base64
from io import BytesIO, TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib import rcParams
from matplotlib import animation

class FuncAnimation(animation.FuncAnimation):

    def to_html5_video(self, embed_limit=None, savefig_kwargs=None):
        """
        Convert the animation to an HTML5 ``<video>`` tag.

        This saves the animation as an h264 video, encoded in base64
        directly into the HTML5 video tag. This respects the rc parameters
        for the writer as well as the bitrate. This also makes use of the
        ``interval`` to control the speed, and uses the ``repeat``
        parameter to decide whether to loop.

        Parameters
        ----------
        embed_limit : float, optional
            Limit, in MB, of the returned animation. No animation is created
            if the limit is exceeded.
            Defaults to :rc:`animation.embed_limit` = 20.0.

        Returns
        -------
        video_tag : str
            An HTML5 video tag with the animation embedded as base64 encoded
            h264 video.
            If the *embed_limit* is exceeded, this returns the string
            "Video too large to embed."
        """
        VIDEO_TAG = r'''<video {size} {options}>
  <source type="video/mp4" src="data:video/mp4;base64,{video}">
  Your browser does not support the video tag.
</video>'''
        # Cache the rendering of the video as HTML
        if not hasattr(self, '_base64_video'):
            # Save embed limit, which is given in MB
            if embed_limit is None:
                embed_limit = rcParams['animation.embed_limit']

            # Convert from MB to bytes
            embed_limit *= 1024 * 1024

            # Can't open a NamedTemporaryFile twice on Windows, so use a
            # TemporaryDirectory instead.
            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir, "temp.m4v")
                # We create a writer manually so that we can get the
                # appropriate size for the tag
                Writer = animation.writers[rcParams['animation.writer']]
                writer = Writer(codec='h264',
                                bitrate=rcParams['animation.bitrate'],
                                fps=1000. / self._interval)
                self.save(str(path), writer=writer, savefig_kwargs=savefig_kwargs)
                # Now open and base64 encode.
                vid64 = base64.encodebytes(path.read_bytes())

            vid_len = len(vid64)
            if vid_len >= embed_limit:
                _log.warning(
                    "Animation movie is %s bytes, exceeding the limit of %s. "
                    "If you're sure you want a large animation embedded, set "
                    "the animation.embed_limit rc parameter to a larger value "
                    "(in MB).", vid_len, embed_limit)
            else:
                self._base64_video = vid64.decode('ascii')
                self._video_size = 'width="{}" height="{}"'.format(
                        *writer.frame_size)

        # If we exceeded the size, this attribute won't exist
        if hasattr(self, '_base64_video'):
            # Default HTML5 options are to autoplay and display video controls
            options = ['controls', 'autoplay']

            # If we're set to repeat, make it loop
            if hasattr(self, 'repeat') and self.repeat:
                options.append('loop')

            return VIDEO_TAG.format(video=self._base64_video,
                                    size=self._video_size,
                                    options=' '.join(options))
        else:
            return 'Video too large to embed.'