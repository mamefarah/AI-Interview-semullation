export enum InterviewState {
  IDLE = 'IDLE',
  CONNECTING = 'CONNECTING',
  IN_CONVERSATION = 'IN_CONVERSATION',
  SUMMARIZING = 'SUMMARIZING',
  FINISHED = 'FINISHED',
  ERROR = 'ERROR'
}

export interface TranscriptEntry {
    speaker: 'Interviewer' | 'You';
    text: string;
}
