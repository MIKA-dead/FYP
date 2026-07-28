import type { Metadata } from "next";
import opencodeConfig from "../../../opencode.json";

export const metadata: Metadata = {
  title: "Agents",
  description: "查看项目中已配置的 OpenCode agents。",
};

export default function AgentsPage() {
  const agents = Object.entries(opencodeConfig.agent ?? {});

  return (
    <div className="min-h-full bg-zinc-50 px-6 py-12 text-slate-950">
      <main className="mx-auto max-w-5xl rounded-3xl border border-zinc-200 bg-white p-10 shadow-sm">
        <h1 className="text-4xl font-semibold tracking-tight">Agents</h1>
        <p className="mt-4 max-w-2xl text-lg leading-8 text-slate-600">
          下面列出了项目中通过 `opencode.json` 注册的 agent，以及它们对应的配置路径。
        </p>

        <div className="mt-10 space-y-5">
          {agents.map(([id, info]) => (
            <div
              key={id}
              className="rounded-3xl border border-zinc-200 bg-slate-50 p-6 shadow-sm"
            >
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <p className="text-2xl font-semibold text-slate-950">{id}</p>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{info.description}</p>
                </div>
                <span className="rounded-full bg-slate-100 px-3 py-1 text-sm font-medium text-slate-700">
                  {info.mode}
                </span>
              </div>
              <div className="mt-5 text-sm leading-6 text-slate-600">
                配置文件：<code className="rounded-md bg-white px-2 py-1 text-sm text-slate-800">.opencode/agents/{id}.md</code>
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
