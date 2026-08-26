import { useState } from "react";
import { AnomaliesTab } from "./components/tabs/AnomaliesTab";
import { ChartsTab } from "./components/tabs/ChartsTab";
import { ChatTab } from "./components/tabs/ChatTab";
import { InsightsTab } from "./components/tabs/InsightsTab";
import { OverviewTab } from "./components/tabs/OverviewTab";
import { ReasoningTab } from "./components/tabs/ReasoningTab";
import { ReportTab } from "./components/tabs/ReportTab";
import { DashboardShell, type TabKey } from "./components/DashboardShell";
import { UploadView } from "./components/UploadView";
import type { DatasetProfile } from "./types";

function App() {
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [tab, setTab] = useState<TabKey>("overview");

  if (!profile) {
    return <UploadView onUploaded={(res) => setProfile(res.profile)} />;
  }

  return (
    <DashboardShell
      profile={profile}
      active={tab}
      onTabChange={setTab}
      onReset={() => {
        setProfile(null);
        setTab("overview");
      }}
    >
      {tab === "overview" && <OverviewTab profile={profile} />}
      {tab === "chat" && <ChatTab profile={profile} />}
      {tab === "reasoning" && <ReasoningTab profile={profile} />}
      {tab === "charts" && <ChartsTab profile={profile} />}
      {tab === "insights" && <InsightsTab profile={profile} />}
      {tab === "anomalies" && <AnomaliesTab profile={profile} />}
      {tab === "report" && <ReportTab profile={profile} />}
    </DashboardShell>
  );
}

export default App;
