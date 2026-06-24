import asyncio

async def do_some_processing():
    await asyncio.sleep(3)
    return "Task 1 Done"

async def do_other_processing():
    await asyncio.sleep(2)
    return "Task 2 Done"

async def do_yet_more_processing():
    await asyncio.sleep(1)
    return "Task 3 Done"

async def main():
    results = await asyncio.gather(
        do_some_processing(),
        do_other_processing(),
        do_yet_more_processing()
    )

    print(results)

asyncio.run(main())