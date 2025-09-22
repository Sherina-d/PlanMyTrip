__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import datetime
import streamlit as st
from dotenv import load_dotenv
import requests

# Updated import - Tool is now imported differently
from crewai import Agent, Task, Crew, LLM
from crewai_tools import tool  # This is the correct import for newer versions

# Alternative import if above doesn't work:
# from crewai.tools import BaseTool

# -----------------------------
# --- Load environment variables ---
# -----------------------------
load_dotenv()

st.set_page_config(
    page_title="🌍 PlanMyTrip",
    page_icon="✈️",
    layout="wide"
)

st.title("🌍 PlanMyTrip")
st.markdown("**🧠 Powered by AI Agents**")

gemini_api_key = os.getenv("GEMINI_API_KEY")

# -----------------------------
# --- DuckDuckGo Search Tool ---
# -----------------------------
@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """
    Retrieve real-time info about destinations, transport, attractions, or hotels
    """
    try:
        res = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=5
        )
        data = res.json()
        results = []
        if "RelatedTopics" in data:
            for item in data["RelatedTopics"][:3]:
                if "Text" in item and "FirstURL" in item:
                    results.append(f"{item['Text']} - {item['FirstURL']}")
        return "\n".join(results) if results else "No live info found."
    except Exception as e:
        return f"Error fetching live info: {str(e)}"

# -----------------------------
# --- Main App Sidebar Inputs ---
# -----------------------------
if gemini_api_key:
    st.sidebar.success("✅ API Key loaded from environment!")

    st.markdown("### 📋 Trip Details")
    col1, col2 = st.columns(2)
    with col1:
        from_city = st.text_input("From City", placeholder="Chennai")
        destination = st.text_input("Destination", placeholder="Goa")
        interests = st.text_input("Interests", placeholder="beaches, nightlife, food")
    with col2:
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        budget_type = st.selectbox("Budget Type", ["budget", "moderate", "luxury"])

    if start_date and end_date:
        if end_date <= start_date:
            st.error("⚠️ End date must be after start date!")
        else:
            trip_duration = (end_date - start_date).days
            st.info(f"Trip Duration: {trip_duration} days")

    # -----------------------------
    # --- Generate Travel Plan ---
    # -----------------------------
    if st.button("🚀 Generate Multi-Agent Travel Plan", type="primary"):
        if from_city and destination and interests and start_date and end_date and end_date > start_date:
            trip_duration = (end_date - start_date).days
            with st.spinner("🤖 Multi-Agent system is working..."):
                try:
                    # -----------------------------
                    # --- Gemini LLM Setup ---
                    # -----------------------------
                    gemini_llm = LLM(
                        model="gemini/gemini-1.5-flash",
                        api_key=gemini_api_key,
                        temperature=0.3
                    )

                    # -----------------------------
                    # --- Agents with Tools ---
                    # -----------------------------
                    transport_agent = Agent(
                        role="Transportation Specialist",
                        goal=f"Find all transportation options from {from_city} to {destination} using real-time info",
                        backstory="Expert in all modes of transport with accurate cost and schedule info",
                        llm=gemini_llm,
                        tools=[duckduckgo_search],  # Use the tool directly
                        verbose=False,
                        allow_delegation=False
                    )

                    stay_agent = Agent(
                        role="Accommodation Specialist",
                        goal=f"Find 5-6 accommodation options in {destination} for {budget_type} travelers using live search",
                        backstory="Hotel and accommodation expert with real-time pricing info",
                        llm=gemini_llm,
                        tools=[duckduckgo_search],
                        verbose=False,
                        allow_delegation=False
                    )

                    itinerary_agent = Agent(
                        role="Itinerary Planning Specialist",
                        goal=f"Create detailed day-wise itinerary for {trip_duration} days in {destination} with real-time info",
                        backstory="Planner who uses live info for attractions, timings, and schedules",
                        llm=gemini_llm,
                        tools=[duckduckgo_search],
                        verbose=False,
                        allow_delegation=False
                    )

                    budget_agent = Agent(
                        role="Budget Analysis Specialist",
                        goal=f"Calculate total trip cost estimation for {budget_type} travel using live info",
                        backstory="Financial expert who uses live info to calculate transportation and accommodation costs",
                        llm=gemini_llm,
                        tools=[duckduckgo_search],
                        verbose=False,
                        allow_delegation=False
                    )

                    coordinator_agent = Agent(
                        role="Travel Plan Coordinator",
                        goal="Merge all agent outputs into one clean, readable final travel plan",
                        backstory="Master coordinator combining all travel info into a comprehensive plan",
                        llm=gemini_llm,
                        verbose=False,
                        allow_delegation=False
                    )

                    transport_task = Task(
                        description=f"""
                        Research transportation options from {from_city} to {destination}.

                        Provide:
                        ## 🚆 Transportation Options

                        ### ✈️ Flight Options
                        - Airlines available
                        - Duration and cost range
                        - Best booking platforms

                        ### 🚂 Train Options  
                        - Train services available
                        - Duration and cost
                        - Booking tips

                        ### 🚌 Bus Options
                        - Bus operators (government/private)
                        - Duration and cost
                        - Comfort levels

                        ### 🚗 Taxi/Car Options
                        - Taxi services
                        - Car rental options
                        - Approximate costs

                        ### 🚇 Local Transport in {destination}
                        - Public transport options
                        - Auto/taxi rates
                        - Transport passes available
                        """,
                        expected_output="Comprehensive transportation guide with all modes of transport and costs",
                        agent=transport_agent
                    )

                    stay_task = Task(
                        description=f"""
                        Find 5-6 accommodation options in {destination} for {budget_type} travelers.

                        Format:
                        ## 🏨 Accommodation Options

                        ### Option 1: [Hotel Name]
                        **Type:** [Hotel/Resort/Guesthouse]
                        **Price Range:** ₹X - ₹Y per night
                        **Location:** [Area/neighborhood]
                        **Rating:** [Star rating or review score]

                        **Perks:**
                        - [Advantage 1]
                        - [Advantage 2]
                        - [Advantage 3]

                        **Cons:**
                        - [Disadvantage 1]
                        - [Disadvantage 2]

                        [Continue for 5-6 options covering different price ranges within {budget_type} category]
                        """,
                        expected_output="5-6 detailed accommodation options with pros/cons, prices, and locations",
                        agent=stay_agent
                    )

                    itinerary_task = Task(
                        description=f"""
                        Create a {trip_duration}-day detailed itinerary for {destination}.
                        Focus on interests: {interests}

                        Format each day:
                        ## Day X: [Date] - [Theme]
                        - 08:00 - 09:00: Breakfast at [place]
                        - 09:00 - 11:00: [Activity with location]
                        - 11:00 - 11:30: Travel/break time
                        - 11:30 - 13:00: [Next activity]
                        - 13:00 - 14:00: Lunch at [restaurant]
                        - 14:00 - 16:00: [Afternoon activity]
                        - 16:00 - 18:00: [Another activity]
                        - 18:00 - 19:00: Rest/return to hotel
                        - 19:00 - 21:00: Dinner at [restaurant]
                        - 21:00 - 22:00: [Evening activity/rest]

                        Include activities matching: {interests}
                        """,
                        expected_output=f"Complete {trip_duration}-day itinerary with time slots and activities",
                        agent=itinerary_agent
                    )

                    budget_task = Task(
                        description=f"""
                        Calculate total cost estimation for this {trip_duration}-day trip to {destination}.
                        Budget category: {budget_type}

                        Calculate costs for:
                        ## 💸 Budget Breakdown

                        ### Transportation Costs
                        - {from_city} to {destination}: ₹X - ₹Y
                        - Local transport: ₹X per day

                        ### Accommodation Costs  
                        - Per night: ₹X - ₹Y
                        - Total ({trip_duration} nights): ₹X - ₹Y

                        ### Food Costs
                        - Breakfast: ₹X per day
                        - Lunch: ₹X per day  
                        - Dinner: ₹X per day
                        - Total food ({trip_duration} days): ₹X - ₹Y

                        ### Activity Costs
                        - Entry fees and activities: ₹X - ₹Y

                        ### Miscellaneous
                        - Shopping, tips, extras: ₹X - ₹Y

                        ## 💰 Final Estimation
                        **Estimated total cost for a {budget_type} trip is ₹X – ₹Y**
                        """,
                        expected_output="Detailed budget breakdown with final cost estimation",
                        agent=budget_agent,
                        context=[transport_task, stay_task, itinerary_task]
                    )

                    coordinator_task = Task(
                        description=f"""
                        Merge all the outputs from transport, accommodation, itinerary, and budget agents into one clean, readable final travel plan.

                        Create a comprehensive plan with:
                        # 🌍 Complete Travel Plan: {from_city} → {destination}

                        [Include all sections from other agents in a well-organized manner]

                        Make it clean, readable, and professional.
                        """,
                        expected_output="Complete, well-organized travel plan combining all agent outputs",
                        agent=coordinator_agent,
                        context=[transport_task, stay_task, itinerary_task, budget_task]
                    )

                    travel_crew = Crew(
                        agents=[transport_agent, stay_agent, itinerary_agent, budget_agent, coordinator_agent],
                        tasks=[transport_task, stay_task, itinerary_task, budget_task, coordinator_task],
                        verbose=False
                    )

                    progress_placeholder = st.empty()

                    with progress_placeholder.container():
                        st.write("🚆 TransportAgent: Finding transportation options...")
                        st.write("🏨 StayAgent: Researching accommodations...")
                        st.write("📅 ItineraryAgent: Creating daily schedule...")
                        st.write("💸 BudgetAgent: Calculating costs...")
                        st.write("🔄 CoordinatorAgent: Merging everything...")

                    result = travel_crew.kickoff()

                    progress_placeholder.empty()

                    st.success("🎉 Multi-Agent Travel Plan Complete!")
                    st.markdown("---")
                    st.markdown("## 🗺️ Your Complete Multi-Agent Travel Plan")
                    st.markdown(str(result))

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"multi_agent_travel_plan_{destination.replace(' ', '_')}_{timestamp}.md"

                    download_content = f"""# 🌍 Multi-Agent AI Travel Plan: {from_city} → {destination}

**Generated by:** 5 Specialized AI Agents
**Date:** {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}
**Trip Dates:** {start_date} to {end_date}
**Duration:** {trip_duration} days
**Interests:** {interests}
**Budget Type:** {budget_type}

---

{str(result)}

---

*🧠 Generated using Multi-Agent Architecture:*
- 🚆 TransportAgent: Transportation options
- 🏨 StayAgent: Accommodation recommendations  
- 📅 ItineraryAgent: Day-wise scheduling
- 💸 BudgetAgent: Cost estimation
- 🔄 CoordinatorAgent: Final plan coordination
"""

                    st.download_button(
                        label="📥 Download Multi-Agent Travel Plan",
                        data=download_content,
                        file_name=filename,
                        mime="text/markdown"
                    )

                    st.info(f"💾 Multi-agent travel plan saved as: {filename}")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Please check your API key and try again")

        else:
            st.warning("⚠️ Please fill in all required fields and ensure end date is after start date!")

else:
    st.error("❌ API Key not found!")
    st.info("Please make sure you have a .env file with GEMINI_API_KEY=your_key")

    st.markdown("### 🧠 Multi-Agent Architecture")
    st.markdown("""
    **🔄 CoordinatorAgent:** Merges outputs from all agents into clean, readable final plan

    **🚆 TransportAgent:** Suggests travel options including bus, train, taxi, and local transport

    **🏨 StayAgent:** Provides 5–6 hotel options based on budget with pros/cons, location, price range

    **📅 ItineraryAgent:** Creates day-wise itinerary with time slots (e.g., 8–9 AM breakfast)

    **💸 BudgetTrackerAgent:** Estimates overall trip cost based on stay type, transport, meals, etc.
    """)
