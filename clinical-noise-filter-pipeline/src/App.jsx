import React, { useState, useRef } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Paper,
  Button,
  LinearProgress,
  Grid,
  Card,
  CardContent,
  Alert,
  Chip,
  Divider,
  ThemeProvider,
  createTheme,
  CssBaseline,
  CircularProgress,
} from '@mui/material';
import MonitorHeart from '@mui/icons-material/MonitorHeart';
import CloudUpload from '@mui/icons-material/CloudUpload';
import PlayArrow from '@mui/icons-material/PlayArrow';
import PictureAsPdf from '@mui/icons-material/PictureAsPdf';
import CheckCircle from '@mui/icons-material/CheckCircle';
import WarningAmber from '@mui/icons-material/WarningAmber';
import Cancel from '@mui/icons-material/Cancel';
import Favorite from '@mui/icons-material/Favorite';
import Timer from '@mui/icons-material/Timer';
import GraphicEq from '@mui/icons-material/GraphicEq';
import InsertDriveFile from '@mui/icons-material/InsertDriveFile';
import ErrorIcon from '@mui/icons-material/Error';
import { uploadEDF, getReportURL } from './api';

// Professional Medical Theme (Clean White & Clinical Blue)
const medicalTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976D2',
      light: '#4791db',
      dark: '#115293',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#0288D1',
    },
    background: {
      default: '#F4F7FB',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#333333',
      secondary: '#666666',
    },
    success: {
      main: '#2E7D32',
      light: '#E8F5E9',
    },
    warning: {
      main: '#ED6C02',
      light: '#FFF3E0',
    },
    error: {
      main: '#D32F2F',
      light: '#FFEBEE',
    },
    info: {
      main: '#0288D1',
      light: '#E1F5FE',
    },
  },
  typography: {
    fontFamily: '"Inter", "Segoe UI", Roboto, Arial, sans-serif',
    h5: {
      fontWeight: 700,
    },
    h6: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 8,
  },
});

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [results, setResults] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith('.edf')) {
        setErrorMsg('Please select a valid European Data Format (.edf) file.');
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
      setErrorMsg(null);
      setResults(null);
    }
  };

  const handleChooseFileClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleRunPipeline = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setErrorMsg(null);
    setResults(null);
    setUploadPercent(0);

    try {
      const response = await uploadEDF(selectedFile, (progressEvent) => {
        if (progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadPercent(percent);
        }
      });
      const data = response?.data || response;

      if (data && data.success) {
        setResults(data);
      } else {
        throw new Error('Pipeline completed without output metrics.');
      }
    } catch (err) {
      console.error('API Error:', err);
      const detail = err.response?.data?.detail || err.message || 'Error processing EDF file.';
      setErrorMsg(detail);
    } finally {
      setIsProcessing(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B';
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(2)} MB`;
  };

  return (
    <ThemeProvider theme={medicalTheme}>
      <CssBaseline />

      {/* Header AppBar */}
      <AppBar position="static" elevation={2} sx={{ backgroundColor: '#1976D2' }}>
        <Toolbar sx={{ py: 1 }}>
          <MonitorHeart sx={{ fontSize: 36, mr: 2, color: '#FFFFFF' }} />
          <Box>
            <Typography variant="h5" component="div" sx={{ fontWeight: 700, color: '#FFFFFF', letterSpacing: 0.5 }}>
              Clinical Noise Filter Pipeline
            </Typography>
            <Typography variant="caption" sx={{ color: '#E3F2FD', fontWeight: 500, display: 'block' }}>
              Automated Holter Triage & Signal Quality Analysis
            </Typography>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Main Content Area */}
      <Container maxWidth="lg" sx={{ py: 4, flex: 1 }}>
        {/* Upload Section */}
        <Paper
          elevation={1}
          sx={{
            p: 4,
            mb: 4,
            border: '2px dashed #B0BEC5',
            backgroundColor: '#FFFFFF',
            borderRadius: 3,
            textAlign: 'center',
            transition: 'border-color 0.2s',
            '&:hover': {
              borderColor: '#1976D2',
            },
          }}
        >
          <input
            type="file"
            accept=".edf"
            ref={fileInputRef}
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />

          <CloudUpload sx={{ fontSize: 56, color: '#1976D2', mb: 1.5 }} />

          <Typography variant="h6" sx={{ color: '#333333', mb: 1, fontWeight: 600 }}>
            Upload Raw Holter ECG (.edf)
          </Typography>
          <Typography variant="body2" sx={{ color: '#666666', mb: 3 }}>
            Upload 12-hour or 24-hour ambulatory ECG recordings for automated triage & artifact filtering
          </Typography>

          <Button
            variant="outlined"
            size="large"
            onClick={handleChooseFileClick}
            disabled={isProcessing}
            sx={{
              px: 4,
              py: 1.2,
              fontWeight: 600,
              textTransform: 'none',
              borderRadius: 2,
              borderColor: '#1976D2',
              borderWidth: 1.5,
              '&:hover': { borderWidth: 1.5 },
            }}
          >
            Choose EDF File
          </Button>

          {/* Selected File Details */}
          {selectedFile && (
            <Box
              sx={{
                mt: 3,
                p: 2,
                display: 'inline-flex',
                alignItems: 'center',
                backgroundColor: '#F8FAFC',
                border: '1px solid #E2E8F0',
                borderRadius: 2,
              }}
            >
              <InsertDriveFile sx={{ color: '#1976D2', mr: 1.5 }} />
              <Box sx={{ textAlign: 'left', mr: 2 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#333333' }}>
                  {selectedFile.name}
                </Typography>
                <Typography variant="caption" sx={{ color: '#666666' }}>
                  Size: {formatFileSize(selectedFile.size)}
                </Typography>
              </Box>
              <Chip label="Ready" size="small" color="primary" sx={{ fontWeight: 600, height: 22 }} />
            </Box>
          )}

          {/* Action Button */}
          <Box sx={{ mt: 3 }}>
            <Button
              variant="contained"
              size="large"
              color="primary"
              disabled={!selectedFile || isProcessing}
              onClick={handleRunPipeline}
              startIcon={isProcessing ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
              sx={{
                px: 5,
                py: 1.4,
                fontSize: '1.05rem',
                fontWeight: 700,
                textTransform: 'none',
                borderRadius: 2,
                boxShadow: 2,
              }}
            >
              {isProcessing ? 'Analyzing ECG...' : 'Run Pipeline'}
            </Button>
          </Box>
        </Paper>

        {/* Loading State Banner */}
        {isProcessing && (
          <Paper elevation={2} sx={{ p: 3, mb: 4, backgroundColor: '#FFFFFF', borderRadius: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1.5 }}>
              <Timer sx={{ color: '#1976D2', mr: 1 }} />
              <Typography variant="subtitle1" sx={{ fontWeight: 600, color: '#1976D2' }}>
                Processing 24-Hour Holter Data... Please wait (approx. 30-60 seconds).
              </Typography>
            </Box>
            <LinearProgress variant="indeterminate" sx={{ height: 8, borderRadius: 4 }} />
            <Typography variant="caption" sx={{ color: '#666666', mt: 1, display: 'block' }}>
              Segmenting 1-second windows, evaluating Lead-Off & high-frequency EMG noise, extracting R-peaks, and rendering Two-Tier PDF.
            </Typography>
          </Paper>
        )}

        {/* Error Notification */}
        {errorMsg && (
          <Alert severity="error" icon={<ErrorIcon />} sx={{ mb: 4, borderRadius: 2 }}>
            {errorMsg}
          </Alert>
        )}

        {/* Results Dashboard */}
        {results && results.metrics && (
          <Box sx={{ mb: 4 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2.5 }}>
              <Box>
                <Typography variant="h5" sx={{ fontWeight: 700, color: '#1976D2' }}>
                  Clinical Triage Results
                </Typography>
                <Typography variant="body2" sx={{ color: '#666666' }}>
                  File: {results.filename} | Duration: {results.duration_hours || '24'} hours ({results.duration_seconds?.toLocaleString()} s) | Total Detected Beats: {results.total_beats?.toLocaleString()}
                </Typography>
              </Box>

              {/* Large Download Button */}
              <Button
                variant="contained"
                color="success"
                size="large"
                startIcon={<PictureAsPdf />}
                href={getReportURL()}
                target="_blank"
                rel="noopener noreferrer"
                sx={{
                  px: 3.5,
                  py: 1.2,
                  fontWeight: 700,
                  fontSize: '0.95rem',
                  textTransform: 'none',
                  borderRadius: 2,
                  boxShadow: 2,
                  backgroundColor: '#2E7D32',
                  '&:hover': {
                    backgroundColor: '#1B5E20',
                  },
                }}
              >
                Download PDF Report
              </Button>
            </Box>

            {/* 4 Classification Cards */}
            <Grid container spacing={2.5}>
              {/* Clean */}
              <Grid item xs={12} sm={6} md={3}>
                <Card
                  elevation={2}
                  sx={{
                    borderRadius: 2,
                    borderTop: '5px solid #2E7D32',
                    backgroundColor: '#FFFFFF',
                  }}
                >
                  <CardContent sx={{ p: 2.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#2E7D32' }}>
                        Clean Rhythm
                      </Typography>
                      <CheckCircle sx={{ color: '#2E7D32', fontSize: 24 }} />
                    </Box>
                    <Typography variant="h3" sx={{ fontWeight: 800, color: '#333333', mb: 0.5 }}>
                      {results.metrics.clean_pct}%
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#666666' }}>
                      Diagnostic Normal HR Beats
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* Bradycardia */}
              <Grid item xs={12} sm={6} md={3}>
                <Card
                  elevation={2}
                  sx={{
                    borderRadius: 2,
                    borderTop: '5px solid #1976D2',
                    backgroundColor: '#FFFFFF',
                  }}
                >
                  <CardContent sx={{ p: 2.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#1976D2' }}>
                        Bradycardia
                      </Typography>
                      <Favorite sx={{ color: '#1976D2', fontSize: 24 }} />
                    </Box>
                    <Typography variant="h3" sx={{ fontWeight: 800, color: '#333333', mb: 0.5 }}>
                      {results.metrics.brady_pct}%
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#666666' }}>
                      Clean Beats with HR &lt; 60 BPM
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* Noise */}
              <Grid item xs={12} sm={6} md={3}>
                <Card
                  elevation={2}
                  sx={{
                    borderRadius: 2,
                    borderTop: '5px solid #ED6C02',
                    backgroundColor: '#FFFFFF',
                  }}
                >
                  <CardContent sx={{ p: 2.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#ED6C02' }}>
                        Noise Artifact
                      </Typography>
                      <WarningAmber sx={{ color: '#ED6C02', fontSize: 24 }} />
                    </Box>
                    <Typography variant="h3" sx={{ fontWeight: 800, color: '#333333', mb: 0.5 }}>
                      {results.metrics.noise_pct}%
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#666666' }}>
                      High-Frequency EMG / Motion
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* Lead-Off */}
              <Grid item xs={12} sm={6} md={3}>
                <Card
                  elevation={2}
                  sx={{
                    borderRadius: 2,
                    borderTop: '5px solid #D32F2F',
                    backgroundColor: '#FFFFFF',
                  }}
                >
                  <CardContent sx={{ p: 2.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#D32F2F' }}>
                        Lead-Off (LD)
                      </Typography>
                      <Cancel sx={{ color: '#D32F2F', fontSize: 24 }} />
                    </Box>
                    <Typography variant="h3" sx={{ fontWeight: 800, color: '#333333', mb: 0.5 }}>
                      {results.metrics.ld_pct}%
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#666666' }}>
                      Electrode Disconnect / Flatline
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Average Heart Rate Summary Bar */}
            <Paper elevation={1} sx={{ mt: 3, p: 3, backgroundColor: '#FFFFFF', borderRadius: 2 }}>
              <Grid container alignItems="center" spacing={2}>
                <Grid item xs={12} sm={6} md={4} sx={{ display: 'flex', alignItems: 'center' }}>
                  <GraphicEq sx={{ color: '#1976D2', fontSize: 32, mr: 1.5 }} />
                  <Box>
                    <Typography variant="caption" sx={{ color: '#666666', fontWeight: 600 }}>
                      AVERAGE HEART RATE (CLEAN)
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 800, color: '#1976D2' }}>
                      {results.metrics.avg_hr} BPM
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6} md={8}>
                  <Typography variant="body2" sx={{ color: '#555555' }}>
                    <strong>Clinical Triage Note:</strong> The pipeline successfully shielded the downstream AI from {results.metrics.ld_pct}% Lead-Off artifacts and {results.metrics.noise_pct}% high-frequency noise, preserving {results.metrics.clean_pct + results.metrics.brady_pct}% diagnostic-quality cardiac rhythm.
                  </Typography>
                </Grid>
              </Grid>
            </Paper>
          </Box>
        )}
      </Container>
    </ThemeProvider>
  );
}
