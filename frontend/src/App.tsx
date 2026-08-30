import { useEffect, useState } from "react";
import {
  DATASET_EXPIRED_MESSAGE,
  markDatasetAvailable,
  subscribeToDatasetExpiry,
} from "./api/client";
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
  const [uploadNotice, setUploadNotice] = useState<string | null>(null);

  useEffect(
    () =>
      subscribeToDatasetExpiry((datasetId) => {
        if (datasetId !== profile?.dataset_id) return;
        setProfile(null);
        setTab("overview");
        setUploadNotice(DATASET_EXPIRED_MESSAGE);
      }),
    [profile?.dataset_id]
  );

  if (!profile) {
    return (
      <UploadView
        notice={uploadNotice}
        onUploaded={(res) => {
          markDatasetAvailable(res.dataset_id);
          setUploadNotice(null);
          setProfile(res.profile);
        }}
      />
    );
  }

  return (
    <DashboardShell
      profile={profile}
      active={tab}
      onTabChange={setTab}
      onReset={() => {
        setProfile(null);
        setTab("overview");
        setUploadNotice(null);
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
