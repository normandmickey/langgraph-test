// agent.ts
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatGroq } from "@langchain/groq";
import { ChatOpenAI } from "@langchain/openai";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import { AskNewsSDK } from '@emergentmethods/asknews-typescript-sdk'
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { DynamicStructuredTool } from "@langchain/core/tools";
import * as dotenv from 'dotenv';
import { SqlToolkit } from "langchain/agents/toolkits/sql";
import { DataSource } from "typeorm";
import { SqlDatabase } from "langchain/sql_db";

dotenv.config();

// IMPORTANT - Add your API keys here. Be careful not to publish them.
const GROQ_API_KEY=process.env.GROQ_API_KEY;
const TAVILY_API_KEY=process.env.TAVILY_API_KEY;
const LANGCHAIN_API_KEY=process.env.LANGCHAIN_API_KEY;
const LANGCHAIN_CALLBACKS_BACKGROUND=process.env.LANGCHAIN_CALLBACKS_BACKGROUND;
const LANGCHAIN_TRACING_V2=process.env.LANGCHAIN_TRACING_V2;
const LANGCHAIN_PROJECT=process.env.LANGCHAIN_PROJECT;

//const agentModel = new ChatGroq({ model: "llama-3.1-70b-Versatile", temperature: 0 });
const agentModel = new ChatOpenAI({ model: "gpt-4o", temperature: 0 });

const datasource = new DataSource({
  type: "sqlite",
  database: "c:/sqlite3/Chinook.db", // Replace with the link to your database
});

const db = await SqlDatabase.fromDataSourceParams({
  appDataSource: datasource,
});

const ask = new AskNewsSDK({
  clientId: process.env.ASKNEWS_CLIENT_ID,
  clientSecret: process.env.ASKNEWS_CLIENT_SECRET,
  scopes: ['news'],
});

async function getNews(query) {
  const response = await ask.news.searchNews({
        query: query, // your keyword query
        nArticles: 5, // control the number of articles to include in the context
        returnType: 'dicts', // you can also ask for "dicts" if you want more information
        method: 'kw', // use "nl" for natural language for your search, or "kw" for keyword search
      });
      return JSON.stringify(response);
}

const toolkit = new SqlToolkit(db, agentModel);
const sqlTools = toolkit.getTools();

// Define the tools for the agent to use
const agentTools = [ 
    new DynamicStructuredTool({
      name: "AskNews",
      description: "Get current news and weather",
      schema: z.object({
        query: z.string().describe('Search Query'),
      }),
      func: async ({query}) => {
        return getNews(query)
      }
    }),
    sqlTools,
];

// Initialize memory to persist state between graph runs
const agentCheckpointer = new MemorySaver();
const agent = createReactAgent({
  llm: agentModel,
  tools: agentTools,
  checkpointSaver: agentCheckpointer,
});

// Now it's time to use!
const agentFinalState = await agent.invoke(
  { messages: [new HumanMessage("what is the five day forecast for Canisteo NY")] },
  { configurable: { thread_id: "42" } },
);

console.log(
  agentFinalState.messages[agentFinalState.messages.length - 1].content,
);

const agentNextState = await agent.invoke(
  { messages: [new HumanMessage("Can you list 10 artists from my database?")] },
  { configurable: { thread_id: "42" } },
);

console.log(
  agentNextState.messages[agentNextState.messages.length - 1].content,
);