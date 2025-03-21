from dotenv import load_dotenv
import os
import refinitiv.data as rd

load_dotenv()  # Loads environment variables from .env
api_key = os.getenv("API_KEY")

# session creation
rd.open_session(app_key=os.getenv("API_KEY"))

# getting ADC and realtime pricing data
universe: str = "SCREEN(U(IN(Equity(private))))"
universe: str = "SCREEN(U(IN(Equity(private))/*UNV:Private*/), IN(TR.HQCountryCode,""AT""), TR.PCFullTimeEmployee(Period=FY0)<250, TR.PCTotRevenueFromBizActv(Period=FY0,Scale=6)<50, CURN=EUR)"
universe: str = "SCREEN(U(IN(Equity(private))/*UNV:Private*/), IN(TR.HQCountryCode,""DE""))"

fields: list[str] = [
    "TR.FULLNAME",
    "TR.CommonName",
    "TR.HeadquartersCountry",
    "TR.CompanyLinkHome",
    "TR.RegistrarWebsite",
    "TR.TotalAssets(Period=FY0, Scale=6)",
    "TR.TotalLiabilities(Period=FY0, Scale=6)",
    "TR.EBIT(Period=FY0, Scale=6)",
    "TR.NetDebt(Period=FY0, Scale=6)",
    "TR.InterestExpense(Period=FY0, Scale=6)",
    "TR.PCTotRevenueFromBizActv(Period=FY0, Scale=6)"
]


df = rd.get_data(universe, fields)

df.to_csv("data/austria.csv")

'@TR("SCREEN(U(IN(Equity(private))/*UNV:Private*/), IN(TR.HQCountryCode,""AT""), TR.PCFullTimeEmployee(Period=FY0)<250, TR.PCTotRevenueFromBizActv(Period=FY0,Scale=6)<50, CURN=EUR,TR.OrganizationStatusCode=Act)";"TR.CommonName;TR.Headquarter"&"sCountry;TR.PCFullTimeEmployee(Period=FY0);TR.PCTotRevenueFromBizActv(Period=FY0,Scale=6)";"curn=EUR RH=In CH=Fd")'

