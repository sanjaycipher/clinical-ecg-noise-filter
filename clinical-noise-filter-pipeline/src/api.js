import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 1200000, // 20 minutes (1,200,000 ms)
});

export const uploadEDF = (file, onProgress) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/api/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress,
  });
};

export const getReportURL = () => 'http://127.0.0.1:8000/api/download';

export default api;
