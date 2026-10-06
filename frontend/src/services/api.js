import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:5000/api",
  timeout: 10000,
});

// Dashboard
export const getDashboardOverview = async () => {
  const response = await api.get("/dashboard/overview");
  return response.data;
};

// Live Traffic
export const getLiveTraffic = async () => {
  const response = await api.get("/traffic/live");
  return response.data;
};

// Detection
export const getLiveDetection = async () => {
  const response = await api.get("/detection/live");
  return response.data;
};

// Attacks
export const getAttacks = async () => {
  const response = await api.get("/attacks");
  return response.data;
};

// Traffic Analytics
export const getTrafficAnalytics = async () => {
  const response = await api.get("/analytics/traffic");
  return response.data;
};

// Model Comparison
export const getModelComparison = async () => {
  const response = await api.get("/models/comparison");
  return response.data;
};

// Mitigation
export const getMitigationStatus = async () => {
  const response = await api.get("/mitigation/status");
  return response.data;
};

export default api;