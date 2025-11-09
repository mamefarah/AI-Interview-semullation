
export const getApiErrorMessage = (error: any, context: 'start' | 'summary'): string => {
    let errorMessage = 'An unknown error occurred.';
    let errorToInspect = error;

    // Handle cases where error is wrapped, e.g. ErrorEvent.error
    if (error?.error) { 
        errorToInspect = error.error;
    }
    
    // Extract a string message from different possible error shapes
    if (errorToInspect instanceof Error) {
        errorMessage = errorToInspect.message;
    } else if (typeof errorToInspect === 'string') {
        errorMessage = errorToInspect;
    } else if (typeof errorToInspect?.message === 'string') {
        errorMessage = errorToInspect.message;
    } else if (typeof error?.message === 'string') { // Fallback to original error object's message
        errorMessage = error.message;
    }

    const lowerCaseMessage = errorMessage.toLowerCase();

    if (lowerCaseMessage.includes('api key not valid')) {
        return 'The provided API key is invalid or not activated for this project. Please check your configuration.';
    }
    if (lowerCaseMessage.includes('permission denied') || lowerCaseMessage.includes('403')) {
        return 'You do not have permission to access this resource. This could be due to an incorrect API key or project settings.';
    }
    if (lowerCaseMessage.includes('quota') || lowerCaseMessage.includes('rate limit') || lowerCaseMessage.includes('429')) {
        return 'The API request quota has been exceeded. Please wait a while before trying again.';
    }
    if (lowerCaseMessage.includes('content has been blocked') || lowerCaseMessage.includes('safety policy')) {
         return 'The request was blocked due to the safety policy. Please modify your script and try again.';
    }
    if (lowerCaseMessage.includes('server error') || lowerCaseMessage.includes('500') || lowerCaseMessage.includes('service unavailable')) {
        return 'The AI service is currently experiencing issues. Please try again later.';
    }
    if (lowerCaseMessage.includes('network error') || lowerCaseMessage.includes('failed to fetch')) {
        return 'A network error occurred. Please check your internet connection and try again.';
    }

    // Context-specific fallbacks using original error message
    if (context === 'start') {
        if (errorMessage === 'An unknown error occurred.') return errorMessage;
        return `Could not start the interview: ${errorMessage}.`;
    }
    if (context === 'summary') {
        if (errorMessage === 'An unknown error occurred.') return errorMessage;
        return `An API error occurred while generating the summary: ${errorMessage}`;
    }

    // Generic fallback
    if (errorMessage === 'An unknown error occurred.') return errorMessage;
    return `An API error occurred: ${errorMessage}`;
};
