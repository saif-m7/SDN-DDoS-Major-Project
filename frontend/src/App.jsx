import { BrowserRouter, Routes, Route } from "react-router-dom";
import Sidebar from "./components/layout/Sidebar";
import Topbar from "./components/layout/Topbar";
import PageContainer from "./components/layout/PageContainer";
import Dashboard from "./pages/Dashboard";
import LiveTraffic from "./pages/LiveTraffic";
import Detection from "./pages/Detection";
import Attacks from "./pages/Attacks";
import Analytics from "./pages/Analytics";
import Mitigation from "./pages/Mitigation";


function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-[#07090d] text-white">
        <Sidebar />

        <div className="ml-[250px]">
          <Topbar />

          <PageContainer>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/traffic" element={<LiveTraffic />} />
              <Route path="/detection" element={<Detection />} />
              <Route path="/attacks" element={<Attacks />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/mitigation" element={<Mitigation />} />
            </Routes>
          </PageContainer>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;