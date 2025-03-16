import io
import sys
from asyncore import write


class StreamToBuffer(io.StringIO):
    """
    A class to redirect the standard output to a buffer and update a placeholder in the interface with the content of the buffer.
    """
    def __init__(self, debug_text=None, progress_text=None, state_text=None, spectra_number_count=None, progress_bar=None, progress_bar_text=None):
        """
        Initialize the StreamToBuffer object.
        :param debug_text: Placeholder to display the debug output.
        :param progress_text: Placeholder to display the progress output for a state.
        :param state_text: Placeholder to display the state output.
        :param spectra_number_count: Placeholder to display the number of spectra output.
        :param progress_bar: Placeholder to display the progress bar.
        """
        super().__init__()
        self.debug_text = debug_text # contains all the output
        self.progress_text = progress_text
        self.state_text = state_text
        self.spectra_number_count = spectra_number_count
        self.spectra_number_count_val = 0
        self.progress_bar = progress_bar
        self.progress_bar_count = progress_bar_text
        self.console = sys.stdout  # store the original stdout for console output


    def write(self, message):
        super().write(message)
        # update the placeholder every time there is new content
        if self.debug_text: self.debug_text.text(f'{self.getvalue()}\n')
        self.console.write(f'{message}\n')

    def write_progress(self, message):
        self.progress_text.empty()
        self.state_text.empty()
        self.progress_text.text(message)
        self.write(f'PROGRESS: {message}')

    def write_state(self, message):
        self.state_text.empty()
        self.state_text.text(message)
        self.write(f'STATE: {message}')

    def write_spectra_number(self, message):
        self.spectra_number_count.empty()
        self.spectra_number_count.text(message)
        self.spectra_number_count_val = int(message)
        self.write(f'SPECTRA NUMBER: {message}')

    def update_progress_bar(self, message):
        self.progress_bar.progress((int(message)+1)/self.spectra_number_count_val)
        self.progress_bar_count.text(f'Predicting spectrum {int(message)+1}/{self.spectra_number_count_val}')

    def flush(self):
        # this ensures that the original console output gets flushed properly
        self.console.flush()

    def close(self):
        """Reset stdout"""
        # Reset sys.stdout to the original standard output (terminal)
        sys.stdout = sys.__stdout__