# Silence Detection Log Feature

## Overview
Added functionality to automatically generate and download detailed silence detection logs showing which sections of the video were detected as silence and removed.

## Changes Made

### 1. Backend: `process_video.py`
- **Added time formatting function**: `format_time()` converts seconds to HH:MM:SS.sss format
- **Save silence detection logs**: After silence detection, a detailed log file is saved with:
  - Summary statistics (total duration, silence segments count, silence removed percentage)
  - **SILENCE SEGMENTS DETECTED**: Lists all detected silence sections with start time, end time, and duration
  - **RETAINED SEGMENTS**: Lists all segments that were kept in the final video with their timestamps
- **Return silence log path**: The `process_video()` function now returns `silence_log_path` in the result dictionary

**Example log format:**
```
================================================================================
SILENCE DETECTION LOG
================================================================================

Total Duration: 00:05:30.123
Total Silence Segments: 5
Total Silence Removed: 35.50%
Original Duration: 00:05:30.123
Processed Duration: 00:03:32.000

--------------------------------------------------------------------------------
SILENCE SEGMENTS DETECTED:
--------------------------------------------------------------------------------

Segment 1:
  From: 00:00:15.500
  To:   00:00:20.300
  Duration: 4.800 seconds

...

--------------------------------------------------------------------------------
RETAINED SEGMENTS:
--------------------------------------------------------------------------------

Segment 1:
  From: 00:00:00.000
  To:   00:00:15.500
  Duration: 15.500 seconds

...
```

### 2. Backend: `app.py`
- **Added silence log download URL** to the response data:
  - Field: `silence_log_file`
  - Format: `/download/{file_id}_silence_log.txt`
- The existing `/download/{filename}` endpoint handles downloading the log file automatically

### 3. Frontend: `index_advanced.html`
- **Added global variable**: `silenceLogUrl` to store the silence log download URL
- **Added button**: "Download Silence Log" button in the video controls section
- **Updated event listeners**:
  - Button click handler downloads the silence log
  - Shows error alert if log is not available
- **Updated result handling**: When processing completes, the silence log URL is stored in the global variable

## Usage

1. **Upload and Process Video**: User uploads a video with noise removal options
2. **Processing Complete**: After processing, a silence detection log is automatically generated
3. **Download Log**: User clicks the "Download Silence Log" button to download the text file
4. **Review Log**: The log file contains detailed information about all detected silence segments and retained segments with precise timestamps

## File Location
- Silence log files are saved in: `backend/outputs/{video_id}_silence_log.txt`
- Format: Plain text file with human-readable formatting

## Benefits
- Users can understand exactly which parts of the video were identified as silence
- Precise timestamps allow users to verify accuracy of silence detection
- Can be used for auditing and quality assurance
- Helpful for adjusting silence detection parameters if needed
